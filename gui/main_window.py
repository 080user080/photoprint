"""
Головне вікно програми PhotoPrint.
Drag & Drop через WM_DROPFILES (utils/win_drop.py) — перевірено на Windows 10/11.

ВИПРАВЛЕННЯ: Зберігаємо базове зображення після перспективної корекції,
щоб слайдери працювали з виправленою перспективою.
"""

import os
import sys
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import cv2

from PyQt6.QtWidgets import QSizePolicy

# OpenCV threading / OpenCL (Task 3)
cv2.setNumThreads(cv2.getNumberOfCPUs())
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QButtonGroup, QRadioButton,
    QFileDialog, QProgressBar, QScrollArea, QApplication,
    QSystemTrayIcon, QMenu, QComboBox, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QPoint
from PyQt6.QtGui import QIcon

# WM_DROPFILES — єдиний надійний механізм Drag&Drop на Windows 10/11 з PyQt6
if sys.platform == "win32":
    from utils.win_drop import register_drop_window, DropEventFilter

from gui.preview         import PreviewPanel
from gui.edit_state      import EditMode, EditSession
from gui.queue_view      import QueueView
from gui.controls        import ControlsPanel
from gui.settings_window import SettingsWindow
from batch.batch_processor import BatchProcessor
from processing import pipeline
from utils import file_utils, image_utils
from utils.coord_transform import image_to_preview_points, preview_to_image_points
from utils.logger import get_logger
from config import app_settings


# Константи
DEFAULT_WINDOW_WIDTH = 1100
DEFAULT_WINDOW_HEIGHT = 680
DEFAULT_QUEUE_WIDTH = 200
MIN_QUEUE_WIDTH = 150
MAX_QUEUE_WIDTH = 500

# Константи для layout
LAYOUT_MARGIN = 8
LAYOUT_SPACING = 8
LEFT_LAYOUT_SPACING = 4
CENTER_LAYOUT_SPACING = 6
CONTROLS_LAYOUT_SPACING = 4
MODE_ROW_SPACING = 8
BUTTONS_ROW_SPACING = 4
BUTTON_HEIGHT = 32

# Константи для таймерів
DROP_SETUP_DELAY_MS = 300

# Константи для режимів
MODE_AUTO_ID = 0
MODE_MANUAL_ID = 1

# Константи для Undo/Redo (TODO1.1)
UNDO_MAX_STEPS = 2

# Константа для стилю статус-бару (щоб уникнути дублювання CSS)
_STATUS_BAR_STYLE = (
    "QStatusBar {"
    "  color: #444444; font-size: 14px;"
    "  background: #D8DCE0;"
    "  border-top: 2px solid #BBBBBB;"
    "  padding: 4px 6px;"
    "  min-height: 30px;"
    "}"
)

_STATUS_BAR_STYLE_SUCCESS = (
    "QStatusBar {"
    "  color: #006600; font-size: 14px;"
    "  background: #D8DCE0;"
    "  border-top: 2px solid #BBBBBB;"
    "  padding: 4px 6px;"
    "  min-height: 30px;"
    "}"
)

_STATUS_BAR_STYLE_ERROR = (
    "QStatusBar {"
    "  color: #CC0000; font-size: 14px;"
    "  background: #D8DCE0;"
    "  border-top: 2px solid #BBBBBB;"
    "  padding: 4px 6px;"
    "  min-height: 30px;"
    "}"
)


# ---------------------------------------------------------------------------
# Worker для авто-режиму (окремий потік — GUI не зависає)
# ---------------------------------------------------------------------------

class AutoWorker(QObject):
    progress = pyqtSignal(int, int, str)        # (1-based index, total, filename) — фаза 1 (обробка)
    print_progress = pyqtSignal(int, int, str) # (1-based index, total, filename) — фаза 2 (друк)
    error    = pyqtSignal(int, str, str)        # (index, filename, message)
    finished = pyqtSignal(int)                  # кількість надрукованих

    def __init__(self, processor: BatchProcessor):
        super().__init__()
        self._p = processor

    def run(self):
        count = self._p.run_auto(
            on_progress=lambda c, t, f: self.progress.emit(c, t, f),
            on_print_progress=lambda c, t, f: self.print_progress.emit(c, t, f),
            on_error=lambda i, f, m: self.error.emit(i, f, m),
        )
        self.finished.emit(count)


# ---------------------------------------------------------------------------
# Worker для однієї операції обробки (фоновий потік)
# ---------------------------------------------------------------------------

class SingleImageWorker(QObject):
    """
    Виконує одну операцію обробки зображення у фоновому потоці.
    func: callable() -> np.ndarray  (або tuple)
    """
    finished = pyqtSignal(object)   # результат операції (np.ndarray або tuple)
    error    = pyqtSignal(str)      # повідомлення про помилку

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CropDetectWorker(QObject):
    """
    TODO2 крок 3.2: легка фонова детекція кутів кадрування.
    На відміну від SingleImageWorker — НЕ блокує кнопки і НЕ показує прогрес-бар.
    func: callable() -> np.ndarray | None
    """
    finished = pyqtSignal(object)   # результат (np.ndarray | None)
    error    = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoPrint")
        self.setMinimumSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self._logger = get_logger(__name__)
        self._settings: Dict[str, Any] = app_settings.load()
        self._processor: BatchProcessor = BatchProcessor(self._settings)
        self._orig: Optional[np.ndarray] = None      # НЕЗМІННИЙ оригінал з диску
        self._base: Optional[np.ndarray] = None      # після autofix + перспективи + авто-корекцій
        self._processed: Optional[np.ndarray] = None # _base + поточні слайдери (фінал для друку)
        # Джерело координат активної crop-сесії. Воно відповідає панелі
        # «До» і не підміняється результатом кадрування між hover-проходами.
        self._crop_source: Optional[np.ndarray] = None
        self._base_for_perspective: Optional[np.ndarray] = None  # знімок _base до початку ручної перспективи
        self._auto_thread: Optional[QThread] = None
        self._single_worker: Optional[SingleImageWorker] = None
        self._single_thread: Optional[QThread] = None
        self._perspective_corners: Optional[np.ndarray] = None  # збережені кути перспективи
        self._perspective_cached_result: Optional[np.ndarray] = None  # кеш perspective для слайдерів
        self._persp_drag_applied: bool = False  # TODO1.5: чи був реальний drag точки (інакше режим скасовуємо без зміни _base)
        self._pending_deskew_result: Optional[np.ndarray] = None      # deskew-результат до коміту (відкладений)
        self._edit_session = EditSession()
        self._drop_filter: Optional[DropEventFilter] = None
        self._current_path: Optional[str] = None  # поточний файл у ручному/перегляді
        self._per_file: Dict[str, Dict[str, Any]] = {}  # збережені налаштування слайдерів по файлу
        # TODO2 крок 3: кеш дефолтної рамки кадрування (прив'язаний до id(self._base))
        self._crop_default_corners_full: Optional[np.ndarray] = None
        self._crop_default_corners_base_id: Optional[int] = None
        self._crop_detection_in_progress: bool = False
        self._crop_detect_thread: Optional[QThread] = None
        self._crop_detect_worker: Optional[CropDetectWorker] = None
        self._crop_detection_base_reference: Optional[np.ndarray] = None
        self._crop_detection_base_id: Optional[int] = None
        self._crop_detection_started: float = 0.0
        # Undo/Redo історія (TODO1.1) — per-file, прив'язана до шляху
        self._undo_history: Dict[str, list] = {}  # path -> list[dict] (знімки _base + _per_file)
        self._redo_history: Dict[str, list] = {}  # path -> list[dict]

        self._settings_win = SettingsWindow()
        self._settings_win.settings_saved.connect(self._on_settings_saved)

        self._build_ui()
        self._apply_default_mode()
        self._apply_preview_colors()
        self._apply_queue_colors()
        self._update_buttons()

        # Завантажуємо розмір вікна та ширину черги
        self._load_window_geometry()

        # Завдання 2.1: Debounce для слайдерів
        self._controls_timer = QTimer()
        self._controls_timer.setSingleShot(True)
        self._controls_timer.setInterval(120)
        self._controls_timer.timeout.connect(self._on_controls_changed_debounced)

        # Завдання 2.2: Відкладене збереження геометрії
        self._save_geometry_timer = QTimer()
        self._save_geometry_timer.setSingleShot(True)
        self._save_geometry_timer.setInterval(1000)
        self._save_geometry_timer.timeout.connect(self._save_window_geometry)

        # Трей-іконка (PRIO 9)
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._setup_tray()

        # Debug dump widget geometries for GUI tester
        if os.environ.get("PHOTOPRINT_DEBUG_WIDGETS") == "1":
            QTimer.singleShot(1500, self._dump_widget_geometries)

        # Drag & Drop реєструємо після показу вікна
        if sys.platform == "win32":
            QTimer.singleShot(DROP_SETUP_DELAY_MS, self._setup_win_drop)

        # Тестові зображення — автозавантаження при старті (за налаштуванням)
        self._load_test_images_if_enabled()

    def _setup_win_drop(self):
        hwnd = int(self.winId())
        register_drop_window(hwnd)
        self._drop_filter = DropEventFilter(self._on_win_drop)
        instance = QApplication.instance()
        if instance is not None:
            instance.installNativeEventFilter(self._drop_filter)

    def _load_test_images_if_enabled(self) -> None:
        """Завантажує тестові зображення в чергу при старті, якщо увімкнено.

        Папка береться з налаштувань `test_images_folder`, або за замовчуванням
        `<корінь проєкту>/tests/test_images`. Якщо папки немає або вона порожня —
        тихо завершується без помилок.
        """
        if not self._settings.get("test_images_enabled", False):
            return
        folder = self._settings.get("test_images_folder", "").strip()
        if not folder:
            folder = str(Path(__file__).resolve().parent.parent / "tests" / "test_images")
        if not os.path.isdir(folder):
            self._logger.warning("Тестова папка не існує: %s", folder)
            return
        imgs = file_utils.collect_images_from_folder(folder)
        if not imgs:
            self._logger.warning("У тестовій папці немає підтримуваних зображень: %s", folder)
            return
        self._queue.add_files(imgs)
        self._on_files_added(imgs)
        self._set_status(f"Завантажено {len(imgs)} тестових зображень")

    def _setup_tray(self):
        """Ініціалізація QSystemTrayIcon (PRIO 9)."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._logger.debug("Трей недоступний на цій системі")
            return
        style = self.style()
        if style is not None:
            icon = style.standardIcon(style.StandardPixmap.SP_ComputerIcon)
        else:
            icon = QIcon()
        self._tray_icon = QSystemTrayIcon(icon, self)
        self._tray_icon.setToolTip("PhotoPrint")

        menu = QMenu()
        act_show = menu.addAction("Відкрити")
        if act_show is not None:
            act_show.triggered.connect(self._show_from_tray)
        menu.addSeparator()
        act_quit = menu.addAction("Вихід")
        if act_quit is not None:
            act_quit.triggered.connect(self._quit_app)

        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()
        self._logger.debug("Трей-іконка створена")

    def _show_from_tray(self):
        """Показує вікно з трею."""
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """Клік по іконці трею — показує/ховає вікно."""
        if reason in (QSystemTrayIcon.ActivationReason.DoubleClick,
                      QSystemTrayIcon.ActivationReason.Trigger):
            if self.isVisible():
                self.hide()
            else:
                self._show_from_tray()

    def _quit_app(self):
        """Повне завершення програми."""
        self._save_window_geometry()
        QApplication.quit()

    def closeEvent(self, event):
        """
        Перевизначення closeEvent (Завд. 2.3):
        - Показуємо QProgressDialog асинхронно, не блокуємо GUI.
        - Якщо minimize_to_tray == True — ховаємо в трей.
        """
        image_utils.preview_cache_clear()
        
        # Завдання 2.3: асинхронне очікування потоків
        if self._has_running_threads():
            from PyQt6.QtWidgets import QProgressDialog
            self._close_progress = QProgressDialog(
                "Завершення обробки...", "Примусово закрити", 0, 0, self
            )
            self._close_progress.setWindowTitle("Завершення роботи")
            self._close_progress.setCancelButtonText("Примусово закрити")
            self._close_progress.canceled.connect(self._force_close)
            self._close_progress.show()
            self._close_start_time = QTimer()  # зберігаємо час початку очікування
            import time
            self._close_start_ms = int(time.time() * 1000)  # ms timestamp
            event.ignore()
            QTimer.singleShot(100, self._check_threads_and_close)
            return
        
        self._do_close(event)

    def _has_running_threads(self) -> bool:
        """Перевіряє чи є активні фонові потоки."""
        auto_running = (hasattr(self, '_auto_thread') and self._auto_thread is not None 
                        and self._auto_thread.isRunning())
        single_running = (hasattr(self, '_single_thread') and self._single_thread is not None 
                          and self._single_thread.isRunning())
        crop_running = (hasattr(self, '_crop_detect_thread') and self._crop_detect_thread is not None
                        and self._crop_detect_thread.isRunning())
        return auto_running or single_running or crop_running

    def _check_threads_and_close(self):
        """Завдання 2.3: перевіряє чи завершились потоки, закриває коли готово."""
        import time
        elapsed = int(time.time() * 1000) - self._close_start_ms
        
        if not self._has_running_threads():
            # Потоки завершились — закриваємо
            if hasattr(self, '_close_progress') and self._close_progress is not None:
                self._close_progress.close()
                self._close_progress = None
            self._do_close(None)
            return
        
        if elapsed > 5000:
            # Минуло >5с — примусово
            self._logger.warning("Завдання 2.3: примусове закриття через 5с таймаут")
            self._force_close()
            return
        
        # Ще чекаємо
        QTimer.singleShot(100, self._check_threads_and_close)

    def _force_close(self):
        """Примусове закриття — зупиняємо потоки та закриваємо."""
        self._logger.info("_force_close: примусове завершення потоків")
        if hasattr(self, '_close_progress') and self._close_progress is not None:
            self._close_progress.close()
            self._close_progress = None
        
        if hasattr(self, '_auto_thread') and self._auto_thread is not None and self._auto_thread.isRunning():
            self._auto_thread.requestInterruption()
            self._auto_thread.quit()
            self._auto_thread.wait(1000)
        if hasattr(self, '_single_thread') and self._single_thread is not None and self._single_thread.isRunning():
            self._cleanup_single_thread()
        if hasattr(self, '_crop_detect_thread') and self._crop_detect_thread is not None:
            self._cleanup_crop_detect_thread()
        
        QApplication.quit()

    def _do_close(self, event):
        """Фактичне закриття — зберігаємо геометрію та закриваємо."""
        if self._settings.get("minimize_to_tray", False) and self._tray_icon is not None:
            self._save_window_geometry()
            self.hide()
            self._tray_icon.showMessage(
                "PhotoPrint",
                "Програму згорнуто в трей. Натисніть на іконку, щоб відкрити.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            if event is not None:
                event.ignore()
        else:
            self._save_window_geometry()
            if event is not None:
                event.accept()
            else:
                QApplication.quit()

    def _wait_for_threads(self):
        """Завдання 2.3: застарілий синхронний метод — тепер не використовується."""
        pass

    def _on_win_drop(self, paths: list[str]):
        """Колбек від WM_DROPFILES — приймає будь-які файли та папки."""
        expanded = []
        for p in paths:
            if os.path.isfile(p):
                expanded.append(p)
            elif os.path.isdir(p):
                expanded.extend(file_utils.collect_images_from_folder(p))
        supported = file_utils.filter_supported(expanded)
        if supported:
            self._queue.add_files(supported)
            self._on_files_added(supported)

    # ------------------------------------------------------------------
    # Побудова UI
    # ------------------------------------------------------------------

    def _apply_background_color(self):
        """Застосовує колір фону центрального віджета з налаштувань."""
        bg_color = self._settings.get("background_color", "#E8E8E8")
        central = self.centralWidget()
        if central is not None:
            central.setStyleSheet(f"background-color: {bg_color};")

    def _apply_preview_colors(self):
        """Застосовує кольори до прев'ю."""
        bg_color = self._settings.get("preview_bg_color", "#E8E8E8")
        text_color = self._settings.get("preview_text_color", "#333333")
        auto_contrast = self._settings.get("auto_contrast_text", True)

        if auto_contrast:
            text_color = self._auto_contrast_color(bg_color)

        self._preview.set_colors(bg_color, text_color)

    def _apply_queue_colors(self):
        """Застосовує кольори до списку черги."""
        bg_color = self._settings.get("queue_bg_color", "#FFFFFF")
        text_color = self._settings.get("queue_text_color", "#111111")
        auto_contrast = self._settings.get("auto_contrast_text", True)

        if auto_contrast:
            text_color = self._auto_contrast_color(bg_color)

        self._queue.set_colors(bg_color, text_color)

    @staticmethod
    def _auto_contrast_color(bg_hex: str) -> str:
        """Автоматично обирає контрастний колір тексту для заданого фону."""
        bg_hex = bg_hex.lstrip("#")
        r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#FFFFFF" if luminance < 128 else "#111111"

    def _build_ui(self):
        central = QWidget()
        bg_color = self._settings.get("background_color", "#E8E8E8")
        central.setStyleSheet(f"background-color: {bg_color};")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN)
        root.setSpacing(LAYOUT_SPACING)

        # QSplitter: ліва панель (черга) + центральна (прев'ю)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("main_splitter")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(6)
        root.addWidget(self._splitter)

        # Ліва панель: черга (обгорнута в QWidget для QSplitter)
        left_widget = QWidget()
        left_widget.setObjectName("left_panel")
        left_widget.setMinimumWidth(MIN_QUEUE_WIDTH)
        left_widget.setMaximumWidth(MAX_QUEUE_WIDTH)
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(LEFT_LAYOUT_SPACING)

        lbl_q = QLabel("Черга файлів")
        lbl_q.setStyleSheet("font-weight:bold; color:#111111; font-size:13px;")
        lbl_q.setObjectName("lbl_queue_title")

        self._queue = QueueView()
        self._queue.setObjectName("queue_view")
        # QueueView.files_dropped — резервний (якщо Qt DnD раптом спрацює)
        self._queue.files_dropped.connect(self._on_win_drop)
        self._queue.selection_changed.connect(self._on_queue_selection)

        btn_add    = QPushButton("Додати файли…")
        btn_add.setObjectName("btn_add_files")
        btn_folder = QPushButton("Додати папку…")
        btn_folder.setObjectName("btn_add_folder")
        btn_clear  = QPushButton("Очистити чергу")
        btn_clear.setObjectName("btn_clear_queue")
        for b in (btn_add, btn_folder, btn_clear):
            b.setStyleSheet(self._btn_style())
        btn_add.clicked.connect(self._browse_files)
        btn_folder.clicked.connect(self._browse_folder)
        btn_clear.clicked.connect(self._clear_queue)

        left.addWidget(lbl_q)
        left.addWidget(self._queue, 1)
        left.addWidget(btn_add)
        left.addWidget(btn_folder)
        left.addWidget(btn_clear)
        self._left_widget = left_widget

        # === Центр: прев'ю + керування внизу (обгорнуто в QWidget для QSplitter) ===
        center_widget = QWidget()
        center_widget.setObjectName("center_panel")
        center_widget.setMinimumWidth(600)
        center = QVBoxLayout(center_widget)
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(CENTER_LAYOUT_SPACING)

        self._preview = PreviewPanel()
        self._preview.perspective_points_changed.connect(self._on_persp_pts_light)
        self._preview.perspective_points_released.connect(self._on_persp_pts_heavy)
        self._preview.crop_session_requested.connect(self._on_crop_session_requested)
        self._preview.crop_session_committed.connect(self._on_crop_session_committed)
        self._preview.crop_rect_released.connect(self._on_crop_preview_changed)
        self._preview.persp_points_changed_hover.connect(self._on_crop_preview_changed)

        self._progress = QProgressBar()
        self._progress.setVisible(False)

        center.addWidget(self._preview, 1)
        center.addWidget(self._progress)
        self._center_widget = center_widget

        # === Внизу під прев'ю: керування ===
        # Панель керування (controls + кнопки)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(CONTROLS_LAYOUT_SPACING)

        # Режим
        mode_row = QHBoxLayout()
        mode_row.setSpacing(MODE_ROW_SPACING)
        lbl_mode = QLabel("Режим:")
        lbl_mode.setStyleSheet("font-weight:bold; color:#111111; font-size:13px;")
        self._radio_auto   = QRadioButton("Пакетний")
        self._radio_auto.setObjectName("radio_auto")
        self._radio_manual = QRadioButton("Покроковий")
        self._radio_manual.setObjectName("radio_manual")
        self._radio_auto.setStyleSheet("color:#111111;")
        self._radio_manual.setStyleSheet("color:#111111;")
        self._radio_auto.setToolTip(
            "Пакетний режим: всі файли з черги будуть автоматично\n"
            "оброблені та надруковані без зупинок.\n"
            "Кнопка «Друкувати все» запускає повну чергу."
        )
        self._radio_manual.setToolTip(
            "Покроковий режим: кожне фото відкривається окремо.\n"
            "Можна відредагувати слайдерами, переглянути результат,\n"
            "потім надрукувати або пропустити."
        )
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self._radio_auto,   MODE_AUTO_ID)
        self._mode_group.addButton(self._radio_manual, MODE_MANUAL_ID)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        mode_row.addWidget(lbl_mode)
        mode_row.addWidget(self._radio_auto)
        mode_row.addWidget(self._radio_manual)
        mode_row.addStretch()

        # Кнопки дій в один ряд
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(BUTTONS_ROW_SPACING)

        self._btn_autofix   = QPushButton("⚡ Auto Fix")
        self._btn_shadow_remove = QPushButton("🌗 Прибрати тінь")
        self._btn_universal = QPushButton("🧩 Універсальна")
        self._btn_print     = QPushButton("🖨  Друк")
        self._btn_skip      = QPushButton("⏭  Пропустити")
        self._btn_print_all = QPushButton("🖨  Друкувати все")
        self._btn_save_img  = QPushButton("💾  Зберегти")
        self._btn_undo      = QPushButton("◀ Назад")
        self._btn_redo      = QPushButton("Вперед ▶")
        self._btn_autofix.setObjectName("btn_autofix")
        self._btn_shadow_remove.setObjectName("btn_shadow_remove")
        self._btn_universal.setObjectName("btn_universal")
        self._btn_print.setObjectName("btn_print")
        self._btn_skip.setObjectName("btn_skip")
        self._btn_print_all.setObjectName("btn_print_all")
        self._btn_save_img.setObjectName("btn_save_image")
        self._btn_undo.setObjectName("btn_undo")
        self._btn_redo.setObjectName("btn_redo")

        for b in (self._btn_autofix, self._btn_shadow_remove, self._btn_universal, self._btn_print,
                   self._btn_skip, self._btn_print_all, self._btn_save_img,
                   self._btn_undo, self._btn_redo):
            b.setFixedHeight(BUTTON_HEIGHT)
            b.setStyleSheet(self._btn_style())

        # Група радіокнопок для режиму тіней
        shadow_group_widget = QWidget()
        shadow_layout = QHBoxLayout(shadow_group_widget)
        shadow_layout.setContentsMargins(4, 0, 4, 0)
        shadow_layout.setSpacing(2)

        shadow_label = QLabel("Тіні:")
        shadow_label.setStyleSheet("color: #555; font-size: 11px;")
        shadow_layout.addWidget(shadow_label)

        self._rb_shadow_auto   = QRadioButton("авто")
        self._rb_shadow_always = QRadioButton("завжди")
        self._rb_shadow_never  = QRadioButton("ніколи")

        for rb in (self._rb_shadow_auto, self._rb_shadow_always, self._rb_shadow_never):
            rb.setStyleSheet("color: #111; font-size: 12px;")
            shadow_layout.addWidget(rb)

        self._rb_shadow_auto.setChecked(True)
        self._rb_shadow_auto.setToolTip(
            "Алгоритм автоматично визначає чи є тіні на документі.\n"
            "Якщо тіней не знайдено — обробка пропускається."
        )
        self._rb_shadow_always.setToolTip(
            "Видаляти тіні завжди, навіть якщо алгоритм не виявив їх.\n"
            "Корисно для документів зі слабкими або нерівномірними тінями."
        )
        self._rb_shadow_never.setToolTip("Не обробляти тіні. Прискорює обробку.")

        self._shadow_mode_group = QButtonGroup()
        self._shadow_mode_group.addButton(self._rb_shadow_auto,   0)
        self._shadow_mode_group.addButton(self._rb_shadow_always, 1)
        self._shadow_mode_group.addButton(self._rb_shadow_never,  2)
        self._shadow_mode_group.idClicked.connect(self._on_shadow_mode_changed)

        self._btn_autofix.clicked.connect(self._do_autofix)
        self._btn_shadow_remove.clicked.connect(self._do_shadow_remove)
        self._btn_universal.clicked.connect(self._do_universal)
        self._btn_print.clicked.connect(self._do_print_current)
        self._btn_skip.clicked.connect(self._do_skip)
        self._btn_print_all.clicked.connect(self._do_print_all)
        self._btn_save_img.clicked.connect(self._do_save_image)
        self._btn_undo.clicked.connect(self._do_undo)
        self._btn_redo.clicked.connect(self._do_redo)

        buttons_row.addWidget(self._btn_autofix)
        buttons_row.addWidget(shadow_group_widget)
        buttons_row.addWidget(self._btn_shadow_remove)
        buttons_row.addWidget(self._btn_universal)
        buttons_row.addWidget(self._btn_print)
        buttons_row.addWidget(self._btn_skip)
        buttons_row.addWidget(self._btn_print_all)
        buttons_row.addWidget(self._btn_save_img)
        buttons_row.addWidget(self._btn_undo)
        buttons_row.addWidget(self._btn_redo)
        buttons_row.addStretch()

        # Налаштування
        btn_settings = QPushButton("⚙  Налаштування")
        btn_settings.setObjectName("btn_settings")
        btn_settings.setStyleSheet(self._btn_style("#555555"))
        btn_settings.clicked.connect(self._open_settings)
        buttons_row.addWidget(btn_settings)

        # Слайдери в два ряди
        self._controls = ControlsPanel()
        self._controls.changed.connect(self._on_controls_changed)
        self._controls.auto_brightness_clicked.connect(self._do_auto_brightness)
        self._controls.auto_contrast_clicked.connect(self._do_auto_contrast)
        self._controls.auto_sharpen_clicked.connect(self._do_auto_sharpen)
        self._controls.perspective_auto_clicked.connect(self._do_persp_auto)
        self._controls.perspective_manual_clicked.connect(self._do_persp_manual)
        self._controls.perspective_reset_clicked.connect(self._do_persp_reset)
        self._controls.reset_all_clicked.connect(self._do_reset_all)

        controls_layout.addLayout(mode_row)
        controls_layout.addLayout(buttons_row)
        controls_layout.addWidget(self._controls)

        # Кнопки "Застосувати deskew" / "Скасувати deskew" — приховані за замовчуванням
        deskew_row = QHBoxLayout()
        deskew_row.setSpacing(8)
        self._btn_apply_deskew = QPushButton("✔ Застосувати deskew")
        self._btn_apply_deskew.setObjectName("btn_apply_deskew")
        self._btn_apply_deskew.setStyleSheet(
            "background:#2E8B57; color:white; border:none; border-radius:4px; padding:5px 12px; font-size:13px; font-weight:bold;"
        )
        self._btn_apply_deskew.setVisible(False)
        self._btn_apply_deskew.clicked.connect(self._commit_deskew)
        self._btn_cancel_deskew = QPushButton("✖ Скасувати deskew")
        self._btn_cancel_deskew.setObjectName("btn_cancel_deskew")
        self._btn_cancel_deskew.setStyleSheet(
            "background:#CD5C5C; color:white; border:none; border-radius:4px; padding:5px 12px; font-size:13px; font-weight:bold;"
        )
        self._btn_cancel_deskew.setVisible(False)
        self._btn_cancel_deskew.clicked.connect(self._cancel_deskew)
        deskew_row.addStretch()
        deskew_row.addWidget(self._btn_apply_deskew)
        deskew_row.addWidget(self._btn_cancel_deskew)
        deskew_row.addStretch()
        controls_layout.addLayout(deskew_row)

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        center.addWidget(controls_container)

        # Додаємо обидві панелі в QSplitter
        self._splitter.addWidget(self._left_widget)
        self._splitter.addWidget(self._center_widget)
        # Встановлюємо початкові пропорції: 200px черга, решта — прев'ю
        self._splitter.setSizes([DEFAULT_QUEUE_WIDTH, max(DEFAULT_WINDOW_WIDTH - DEFAULT_QUEUE_WIDTH - LAYOUT_MARGIN * 2 - LAYOUT_SPACING, 600)])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        # Статусний рядок вбудований у QMainWindow
        sb = self.statusBar()
        if sb is not None:
            sb.setStyleSheet(_STATUS_BAR_STYLE)

            self._status_file_label = QLabel("")
            self._status_file_label.setStyleSheet("color: #888888; font-size: 12px; padding-right: 8px;")
            sb.addPermanentWidget(self._status_file_label)

    def _btn_style(self, color="#2E5FA3"):
        return (
            f"QPushButton{{background:{color};color:white;border:none;"
            f"border-radius:4px;padding:5px 10px;font-size:13px;}}"
            f"QPushButton:hover{{background:{color}DD;}}"
            f"QPushButton:pressed{{background:{color}99;}}"
            f"QPushButton:disabled{{background:#AAAAAA;color:#EEEEEE;}}"
        )

    # ------------------------------------------------------------------
    # Налаштування
    # ------------------------------------------------------------------

    def _on_mode_changed(self, btn_id: int):
        if btn_id == MODE_AUTO_ID:
            self._set_status(
                "Пакетний режим: натисніть «Друкувати все» щоб обробити всю чергу"
            )
        else:
            self._set_status(
                "Покроковий режим: редагуйте кожне фото та натискайте «Друк»"
            )
        self._update_buttons()

    def _apply_default_mode(self):
        if self._settings.get("default_mode", "auto") == "auto":
            self._radio_auto.setChecked(True)
        else:
            self._radio_manual.setChecked(True)
        self._controls.set_shadow_highlight(self._settings.get("shadow_highlight_strength", 0.0))
        self._controls.set_sharpen(self._settings.get("sharpen_strength", 0.4))
        self._controls.set_hdr(self._settings.get("hdr_strength", 0.0))
        # Ініціалізація радіокнопок режиму видалення тіней
        shadow_mode = self._settings.get("shadow_remove_mode", "auto")
        mode_to_btn = {
            "auto": self._rb_shadow_auto,
            "always": self._rb_shadow_always,
            "never": self._rb_shadow_never,
        }
        btn = mode_to_btn.get(shadow_mode, self._rb_shadow_auto)
        btn.setChecked(True)

    def _load_window_geometry(self):
        """Завантажує розмір вікна та ширину черги з налаштувань."""
        width = self._settings.get("window_width", 1100)
        height = self._settings.get("window_height", 680)
        queue_width = self._settings.get("queue_width", 200)
        self.resize(width, height)
        # Відновлюємо розмір панелей QSplitter (замість setFixedWidth)
        if hasattr(self, '_splitter'):
            center_width = max(width - queue_width - LAYOUT_MARGIN * 2 - LAYOUT_SPACING, 600)
            self._splitter.setSizes([queue_width, center_width])

    def _save_window_geometry(self):
        """Зберігає розмір вікна та ширину черги в налаштуваннях."""
        self._settings["window_width"] = self.width()
        self._settings["window_height"] = self.height()
        if hasattr(self, '_splitter'):
            sizes = self._splitter.sizes()
            if len(sizes) >= 2:
                self._settings["queue_width"] = sizes[0]
        else:
            self._settings["queue_width"] = self._queue.width()
        app_settings.save(self._settings)

    def resizeEvent(self, event):
        """Перевизначення resizeEvent — відкладене збереження через таймер (Завд. 2.2)."""
        super().resizeEvent(event)
        self._save_geometry_timer.start()

    def keyPressEvent(self, event):
        """Гарячі клавіші Undo/Redo (TODO1.1): Ctrl+Z / Ctrl+Y."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Z:
                self._do_undo()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Y:
                self._do_redo()
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_shadow_mode_changed(self, btn_id: int):
        """Зміна режиму видалення тіней через радіокнопки."""
        mode_map = {0: "auto", 1: "always", 2: "never"}
        mode = mode_map.get(btn_id, "auto")
        self._settings["shadow_remove_mode"] = mode
        app_settings.save(self._settings)
        # Перезапускаємо Auto Fix з новим режимом
        self._do_autofix_classic()

    def _on_settings_saved(self, s: dict):
        self._settings = s
        self._processor = BatchProcessor(self._settings)
        self._sync_processor_with_queue()
        self._apply_default_mode()
        self._apply_background_color()
        self._apply_preview_colors()
        self._apply_queue_colors()

    def _open_settings(self):
        self._settings_win.load_from_file()
        self._settings_win.show()
        self._settings_win.raise_()

    # ------------------------------------------------------------------
    # Черга
    # ------------------------------------------------------------------

    def _sync_processor_with_queue(self):
        """Завдання 2.5: синхронізує _processor з _queue."""
        self._processor.set_files(self._queue.get_all_paths())

    def _on_files_added(self, paths: list[str]):
        """Спільний обробник після додавання файлів будь-яким способом."""
        supported = file_utils.filter_supported(paths)
        if not supported:
            self._set_status("Жоден з файлів не підтримується")
            return
        was_empty = self._processor.total == 0
        # Завдання 2.5: не оновлюємо _processor тут, тільки чергу
        self._set_status(
            f"Додано {len(supported)} файл(ів). Всього у черзі: {self._processor.total}"
        )
        self._update_buttons()
        # Автоматично відкриваємо перший файл якщо черга була порожня
        if was_empty and supported:
            self._on_queue_selection(supported[0])

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Оберіть зображення", "",
            "Зображення (*.jpg *.jpeg *.png *.webp *.tiff *.tif *.heic *.heif)"
        )
        if paths:
            self._queue.add_files(paths)
            self._on_files_added(paths)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Оберіть папку")
        if folder:
            imgs = file_utils.collect_images_from_folder(folder)
            if imgs:
                self._queue.add_files(imgs)
                self._on_files_added(imgs)
            else:
                self._set_status("У папці немає підтримуваних зображень")

    def _clear_queue(self):
        self._queue.clear_queue()
        self._processor.clear()
        self._preview.clear()
        self._orig = None
        self._base = None
        self._crop_source = None
        self._invalidate_crop_default_cache()  # TODO2 крок 3.3
        self._base_for_perspective = None
        self._processed = None
        self._perspective_corners = None  # скидаємо кути перспективи
        self._perspective_cached_result = None
        self._pending_deskew_result = None
        self._clear_edit_mode()
        self._preview.disable_perspective_edit()
        self._current_path = None
        self._per_file.clear()
        # TODO1.1: очищуємо undo/redo історію при очищенні черги
        self._undo_history.clear()
        self._redo_history.clear()
        image_utils.preview_cache_clear()
        self._update_buttons()
        self._set_status("Черга очищена")

    def _store_current_settings(self):
        """Зберігає поточні значення слайдерів для _current_path."""
        path = self._current_path
        if path is None:
            return
        self._per_file[path] = self._controls.values()

    def _restore_file_settings(self, path: str):
        """Відновлює слайдери для файлу path, або скидає до дефолту (без сигналів)."""
        vals = self._per_file.get(path)
        if vals is None:
            # Скидаємо всі слайдери в дефолтні значення silent — без сигналів
            self._controls.set_shadow_highlight(0.0, silent=True)
            self._controls.set_brightness(0.0, silent=True)
            self._controls.set_contrast(0.0, silent=True)
            self._controls.set_sharpen(0.4, silent=True)
            self._controls.set_hdr(0.0, silent=True)
            self._controls.set_grayscale(False, silent=True)
            return
        self._controls.set_shadow_highlight(vals.get("shadow_highlight", 0.0), silent=True)
        self._controls.set_brightness(vals.get("brightness", 0.0), silent=True)
        self._controls.set_contrast(vals.get("contrast", 0.0), silent=True)
        self._controls.set_sharpen(vals.get("sharpen_strength", 0.0), silent=True)
        self._controls.set_hdr(vals.get("hdr_strength", 0.0), silent=True)
        self._controls.set_grayscale(vals.get("grayscale", False), silent=True)
        # Не викликаємо _on_controls_changed тут - дозволяємо слайдерам працювати самостійно

    # ------------------------------------------------------------------
    # Undo/Redo історія (TODO1.1)
    # ------------------------------------------------------------------

    def _push_undo_snapshot(self) -> None:
        """Зберігає поточний стан (_base + _per_file) в undo-стек для _current_path.
        Очищує redo-стек (будь-яка нова дія користувача скидає redo)."""
        path = self._current_path
        if path is None or self._base is None:
            return
        snapshot = {
            "base": self._base.copy(),
            "per_file": dict(self._per_file.get(path, {})),
            "autofix_applied": self._preview.get_autofix_applied(),
        }
        undo = self._undo_history.setdefault(path, [])
        undo.append(snapshot)
        # Обмежуємо розмір undo-стеку
        if len(undo) > UNDO_MAX_STEPS:
            del undo[0]
        # Нова дія очищує redo-стек
        self._redo_history.pop(path, None)

    def _set_edit_mode(self, mode: EditMode) -> None:
        """Оновлює явний режим; legacy-поля поки лишаються storage."""
        # PyQt забороняє читати атрибути QObject у тестовій заглушці,
        # створеній через ``__new__`` без виклику конструктора.
        try:
            session = getattr(self, "_edit_session", None)
        except RuntimeError:
            return
        if session is not None:
            session.mode = mode

    def _clear_edit_mode(self) -> None:
        self._set_edit_mode(EditMode.IDLE)

    def _commit_base_result(
        self,
        result: np.ndarray,
        status_msg: str | None = None,
        autofix_applied: str | None = None,
        *,
        push_undo: bool = True,
        update_before: bool = False,
        update_after: bool = True,
        update_autofix: bool = False,
        update_buttons: bool = True,
    ) -> None:
        """Єдина точка коміту результату в `_base`.

        Тут зібрані тільки спільні інваріанти: один undo-знімок, оновлення
        `_base`/`_processed`, інвалідація crop-кешу та синхронізація preview.
        Специфічний стан ручної перспективи або deskew кожен caller завершує
        окремо, тому helper не приховує переходи між режимами.
        """
        if push_undo:
            self._push_undo_snapshot()
        self._base = result.copy()
        self._processed = result.copy()
        self._invalidate_crop_default_cache()
        if update_before:
            self._preview.set_before(image_utils.make_preview(self._base))
        if update_after:
            self._preview.set_after(image_utils.make_preview(self._processed))
        if update_autofix:
            self._preview.set_autofix_applied(autofix_applied)
        if update_buttons:
            self._update_buttons()
        if status_msg:
            self._set_status(status_msg)

    def _apply_history_state(self, state: dict) -> None:
        """Застосовує збережений стан до _base та _per_file."""
        if state is None:
            return
        base_img: Optional[np.ndarray] = state.get("base")
        if base_img is None:
            return
        new_base = base_img.copy()
        self._base = new_base
        self._crop_source = None
        self._invalidate_crop_default_cache()  # TODO2 крок 3.3
        path = self._current_path
        if path is not None:
            self._per_file[path] = dict(state.get("per_file", {}))
            self._restore_file_settings(path)
        self._processed = new_base.copy()
        self._base_for_perspective = None
        self._perspective_corners = None
        self._perspective_cached_result = None
        self._pending_deskew_result = None
        self._clear_edit_mode()
        self._preview.disable_perspective_edit()
        self._unfreeze_preview_panels()
        self._preview.set_before(image_utils.make_preview(new_base))
        self._preview.set_after(image_utils.make_preview(new_base))
        # TODO1.1: відновлюємо індикатор Auto Fix із знімку (якщо збережено)
        self._preview.set_autofix_applied(state.get("autofix_applied"))
        self._update_buttons()

    def _do_undo(self) -> None:
        """Скасовує останню дію (Ctrl+Z / кнопка «◀ Назад»)."""
        if self._has_running_threads():
            return
        path = self._current_path
        if path is None:
            return
        undo = self._undo_history.get(path, [])
        if not undo:
            self._set_status("Немає дій для скасування")
            return
        # Поточний стан зберігаємо в redo-стек
        current = {
            "base": self._base.copy() if self._base is not None else None,
            "per_file": dict(self._per_file.get(path, {})),
            "autofix_applied": self._preview.get_autofix_applied(),
        }
        redo = self._redo_history.setdefault(path, [])
        redo.append(current)
        if len(redo) > UNDO_MAX_STEPS:
            del redo[0]
        # Відновлюємо попередній стан
        state = undo.pop()
        self._apply_history_state(state)
        self._set_status("Скасовано (Undo)")

    def _do_redo(self) -> None:
        """Повторює скасовану дію (Ctrl+Y / кнопка «Вперед ▶»)."""
        if self._has_running_threads():
            return
        path = self._current_path
        if path is None:
            return
        redo = self._redo_history.get(path, [])
        if not redo:
            self._set_status("Немає дій для повторення")
            return
        # Поточний стан зберігаємо в undo-стек
        current = {
            "base": self._base.copy() if self._base is not None else None,
            "per_file": dict(self._per_file.get(path, {})),
            "autofix_applied": self._preview.get_autofix_applied(),
        }
        undo = self._undo_history.setdefault(path, [])
        undo.append(current)
        if len(undo) > UNDO_MAX_STEPS:
            del undo[0]
        # Відновлюємо наступний стан
        state = redo.pop()
        self._apply_history_state(state)
        self._set_status("Повторено (Redo)")

    def _finish_perspective_session(self, commit: bool) -> None:
        """TODO1.5: завершує сесію ручної перспективи.
        Якщо commit=True — комітить поточні _perspective_corners у _base.
        У будь-якому разі виходить із режиму (скидає _base_for_perspective/
        _perspective_corners/кеш/драг-флаг і розморожує панелі)."""
        if self._base_for_perspective is None or self._perspective_corners is None:
            return
        if commit:
            result = pipeline.run_perspective_manual(
                self._base_for_perspective, self._perspective_corners
            )
            # Undo вже створює `_commit_pending_perspective`; тут лише
            # застосовуємо результат через спільний commit-шлях.
            self._commit_base_result(
                result,
                push_undo=False,
                update_before=False,
                update_after=False,
                update_buttons=False,
            )
        self._base_for_perspective = None
        self._perspective_corners = None
        self._perspective_cached_result = None
        self._persp_drag_applied = False
        self._clear_edit_mode()
        self._preview.disable_perspective_edit()
        self._unfreeze_preview_panels()

    def _commit_pending_perspective(self, commit: bool = True) -> None:
        """TODO1.5: комітить незавершену ручну перспективу в _base ТІЛЬКИ якщо
        був реальний drag точки. Інакше (простий вхід у режим без перетягування)
        просто виходить з режиму, не змінюючи _base/orig."""
        if self._base_for_perspective is None or self._perspective_corners is None:
            return
        really_commit = commit and self._persp_drag_applied
        if really_commit:
            self._push_undo_snapshot()
        self._finish_perspective_session(commit=really_commit)

    def _on_queue_selection(self, path: str):
        """Клік на файл у списку — завантажуємо для перегляду."""
        try:
            # Завдання 2.1: зупиняємо попередній фоновий потік, якщо ще виконується
            if self._single_thread is not None and self._single_thread.isRunning():
                self._logger.warning("_on_queue_selection: попередній потік ще виконується — очищуємо")
                self._cleanup_single_thread()
            # Завдання 2.2: очищуємо кеш прев'ю перед завантаженням нового файлу
            self._show_deskew_buttons(False)
            image_utils.preview_cache_clear()
            # Скидаємо режим редагування перспективи для попереднього зображення
            self._preview.disable_perspective_edit()
            self._store_current_settings()
            from core import loader
            img = loader.load(path)
            self._orig = img
            self._base = img.copy()  # початково base = orig
            self._crop_source = None
            self._invalidate_crop_default_cache()  # TODO2 крок 3.3
            self._base_for_perspective = None
            self._processed = None
            self._perspective_corners = None  # скидаємо кути перспективи для нового файлу
            self._current_path = path
            self._restore_file_settings(path)
            prev = image_utils.make_preview(img)
            self._preview.set_before(prev)
            # Авто-застосування Auto Fix при завантаженні
            if self._settings.get("auto_apply_autofix", True):
                self._do_autofix_classic()
            else:
                self._update_buttons()
        except Exception as e:
            self._logger.error(f"Помилка завантаження файлу {path}: {e}", exc_info=True)
            self._set_status(f"Помилка завантаження: {e}")

    # ------------------------------------------------------------------
    # Обробка зображень
    # ------------------------------------------------------------------

    def _do_autofix(self):
        self._do_autofix_classic()

    def _compute_full_autofix(self, base_snapshot: np.ndarray, vals: dict):
        """Обчислює повний Auto Fix без змін GUI-стану."""
        s = self._settings
        if s.get("autofix_enabled", True):
            result, status_msg, log_entries = pipeline.run_autofix(
                base_snapshot,
                sharpen_strength=vals["sharpen_strength"],
                hdr_strength=vals["hdr_strength"],
                use_hdr=s.get("hdr_in_autofix", True),
                use_perspective=s.get("auto_perspective", False),
                partial_perspective=s.get("partial_perspective", False),
                bw_binary=s.get("bw_binary", False),
                classify_bw_std_thresh=s.get("classify_bw_std_thresh", 20.0),
                classify_edge_ratio_min=s.get("classify_edge_ratio_min", 0.03),
                classify_line_count_min=s.get("classify_line_count_min", 3),
                shadow_highlight_strength=vals["shadow_highlight"],
                output_color_mode=s.get("output_color_mode", "auto"),
                autofix_contrast=s.get("autofix_contrast", 0.15),
                contrast_mode=s.get("contrast_mode", "linear"),
                settings=s,
            )
            preset = s.get("pipeline_preset", "doc_bw")
            if preset == "custom":
                enabled = {item.strip() for item in s.get("pipeline_steps_enabled", "").split(",") if item.strip()}
                brightness_enabled = "brightness" in enabled
            else:
                brightness_enabled = preset in ("doc_bw", "doc_color")
            if s.get("auto_brightness_enabled", False) and brightness_enabled:
                result = pipeline.run_auto_brightness(
                    result,
                    percentile_low=s.get("auto_percentile_low", 5.0),
                    percentile_high=s.get("auto_percentile_high", 95.0),
                )
                log_entries.append({"step": "auto_brightness", "applied": True,
                                    "detail": "авто-яскравість"})
            if vals["grayscale"]:
                result = pipeline.run_grayscale(result)
            return result, status_msg, log_entries

        result = pipeline.run_manual_adjustments(
            base_snapshot,
            brightness=vals["brightness"],
            contrast=vals["contrast"],
            sharpen_strength=vals["sharpen_strength"],
            hdr_strength=vals["hdr_strength"],
            grayscale=vals["grayscale"],
            shadow_highlight_strength=vals["shadow_highlight"],
            contrast_mode=s.get("contrast_mode", "linear"),
        )
        return result, "Ручні налаштування", []

    def _do_autofix_classic(self):
        if self._orig is None:
            self._set_status("Спочатку оберіть файл")
            return
        self._clear_edit_mode()
        if self._single_thread is not None and self._single_thread.isRunning():
            self._logger.warning("AutoFix: попередній потік ще виконується — чекаємо завершення")
            self._cleanup_single_thread()

        # Commit відкладеного deskew перед autofix
        if self._pending_deskew_result is not None:
            self._commit_deskew()
        # Commit перспективи перед autofix
        if self._base_for_perspective is not None and self._perspective_corners is not None:
            self._base = pipeline.run_perspective_manual(
                self._base_for_perspective, self._perspective_corners
            )
            self._base_for_perspective = None
            self._perspective_corners = None
            self._perspective_cached_result = None
            self._preview.disable_perspective_edit()

        vals = self._controls.values()
        if self._base is None:
            self._set_status("Немає базового зображення для обробки")
            return
        base_snapshot = self._base.copy()   # знімок щоб не передавати self в інший потік
        # Завдання 2.3: зберігаємо _current_path на момент запуску
        path_snapshot = self._current_path

        def _work():
            return self._compute_full_autofix(base_snapshot, vals)

        def _on_done(payload):
            result, status_msg, log_entries = payload
            # Завдання 2.3: ігноруємо результат, якщо файл вже змінився
            if self._current_path != path_snapshot:
                self._logger.debug("_do_autofix_classic: файл змінився, ігноруємо застарілий результат")
                return
            # Побудова розгорнутого рядка з log_entries
            if log_entries:
                # Відфільтровуємо тільки applied кроки
                applied = [e for e in log_entries if e.get("applied")]
                # Виокремлюємо doc_type та color_mode на початок
                type_parts = [e["detail"] for e in applied if e["step"] in ("doc_type", "color_mode")]
                other_parts = [e["detail"] for e in applied if e["step"] not in ("doc_type", "color_mode")]
                all_parts = type_parts + other_parts
                detailed_status = "Auto Fix: " + " | ".join(all_parts)
            else:
                detailed_status = status_msg
            self._commit_base_result(
                result,
                detailed_status,
                "auto_fix",
                update_autofix=True,
            )
            self._set_status(detailed_status)
            # Скидаємо стиль через таймер (1.5с)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: self._reset_status_style())
            self._update_buttons()

        self._run_in_background(_work, _on_done, button_to_lock=self._btn_autofix)

    def _on_controls_changed(self, vals: Optional[dict] = None):
        """Завдання 2.1: Debounce — тільки запускає таймер, не обробляє негайно.
        При reset_all викликається безпосередньо."""
        self._controls_timer.start()

    def _on_controls_changed_debounced(self, vals: Optional[dict] = None):
        """Миттєво оновлює прев'ю при зміні будь-якого слайдера."""
        if self._base is None and self._base_for_perspective is None:
            return
        try:
            vals = self._controls.values()

            # Визначаємо базу з урахуванням активної перспективи
            if (self._base_for_perspective is not None
                    and self._perspective_corners is not None):
                # Використовуємо кешований результат перспективи, якщо є
                base_for_sliders = self._perspective_cached_result
                if base_for_sliders is None:
                    base_for_sliders = pipeline.run_perspective_manual(
                        self._base_for_perspective, self._perspective_corners
                    )
                    self._perspective_cached_result = base_for_sliders
            else:
                base_for_sliders = self._base

            if base_for_sliders is None:
                return

            result = pipeline.run_manual_adjustments(
                base_for_sliders,
                brightness=vals["brightness"],
                contrast=vals["contrast"],
                sharpen_strength=vals["sharpen_strength"],
                hdr_strength=vals["hdr_strength"],
                grayscale=vals["grayscale"],
                shadow_highlight_strength=vals["shadow_highlight"],
                contrast_mode=self._settings.get("contrast_mode", "linear"),
            )
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            # Скидаємо індикатор Auto Fix при ручних налаштуваннях
            self._preview.set_autofix_applied(None)
            self._update_buttons()
            # Зберігаємо нові значення для поточного файлу
            self._store_current_settings()
        except Exception as e:
            self._logger.error(f"Помилка обробки слайдерів: {e}", exc_info=True)
            self._set_status(f"Помилка обробки: {e}")

    def _do_auto_brightness(self):
        if self._base is None:
            return
        s = self._settings
        base_snapshot = self._base.copy()

        def _work():
            return pipeline.run_auto_brightness(
                base_snapshot,
                percentile_low=s.get("auto_percentile_low", 5.0),
                percentile_high=s.get("auto_percentile_high", 95.0),
            )

        def _on_done(result):
            self._perspective_corners = None
            self._commit_base_result(result, "Авто-яскравість застосована")

        self._run_in_background(_work, _on_done)

    def _do_auto_contrast(self):
        if self._base is None:
            return
        s = self._settings
        base_snapshot = self._base.copy()

        def _work():
            return pipeline.run_auto_contrast(
                base_snapshot,
                percentile_low=s.get("auto_percentile_low", 5.0),
                percentile_high=s.get("auto_percentile_high", 95.0),
            )

        def _on_done(result):
            self._perspective_corners = None
            self._commit_base_result(result, "Авто-контраст застосований")

        self._run_in_background(_work, _on_done)

    def _do_auto_sharpen(self):
        if self._base is None:
            return
        s = self._settings
        base_snapshot = self._base.copy()

        def _work():
            return pipeline.run_auto_sharpen(
                base_snapshot,
                threshold=s.get("autosharp_threshold", 80.0),
                max_strength=s.get("autosharp_max_strength", 0.7),
            )

        def _on_done(result):
            result, strength = result
            self._controls.set_sharpen(strength)
            if strength > 0:
                status = f"Авто-різкість застосована ({strength:.2f})"
            else:
                status = "Зображення достатньо різке — різкість не потрібна"
            self._commit_base_result(result, status)

        self._run_in_background(_work, _on_done)

    def _do_shadow_remove(self):
        """Застосувати лише форсоване видалення тіні до поточного зображення.

        Використовується Варіант A: ця кнопка ігнорує режим авто/завжди/ніколи
        та напряму запускає ручний pipeline-крок видалення тіні.
        """
        if self._base is None:
            if self._orig is None:
                self._set_status("Спочатку оберіть файл")
            return

        settings = dict(self._settings)
        base_snapshot = self._base.copy()

        def _work():
            doc_type = pipeline.run_classify(
                base_snapshot,
                bw_std_thresh=settings.get("classify_bw_std_thresh", 20.0),
                edge_ratio_min=settings.get("classify_edge_ratio_min", 0.03),
                line_count_min=settings.get("classify_line_count_min", 3),
            )
            is_color_document = doc_type == pipeline.DocType.COLOR_DOCUMENT.value
            return pipeline.run_shadow_remove_manual(
                base_snapshot,
                is_color_document,
                settings,
            )

        def _on_done(payload):
            result, _had_shadow = payload
            self._perspective_corners = None
            self._commit_base_result(
                result,
                "Видалення тіні застосовано",
                autofix_applied=None,
                update_before=False,
                update_after=True,
            )

        self._run_in_background(
            _work,
            _on_done,
            button_to_lock=self._btn_shadow_remove,
        )

    def _do_universal(self):
        """Застосувати налаштований набір кроків однією undo-операцією."""
        if self._base is None:
            if self._orig is None:
                self._set_status("Спочатку оберіть файл")
            return

        settings = dict(self._settings)
        base_snapshot = self._base.copy()

        def _work():
            return pipeline.run_universal(base_snapshot, settings)

        def _on_done(payload):
            result, status_msg = payload
            if status_msg == "Не обрано жодного кроку":
                self._set_status(
                    "Універсальна кнопка: не обрано жодного кроку в Налаштуваннях"
                )
                return

            self._perspective_corners = None
            self._commit_base_result(
                result,
                status_msg,
                autofix_applied=None,
                update_before=False,
                update_after=True,
            )

        self._run_in_background(
            _work,
            _on_done,
            button_to_lock=self._btn_universal,
        )

    def _do_persp_auto(self):
        # Завдання 3.5: фіксуємо розмір панелей (буде розморожено при скиданні перспективи)
        """Авто-детекція перспективи: коміт попередньої, знімок, авто, показ результатів."""
        if self._orig is None or self._base is None:
            return
        self._set_edit_mode(EditMode.PERSPECTIVE)
        if self._single_thread is not None and self._single_thread.isRunning():
            self._set_status("⏳ Зачекайте, операція ще виконується")
            return

        # Commit попередньої перспективи якщо була
        if self._base_for_perspective is not None and self._perspective_corners is not None:
            self._base = pipeline.run_perspective_manual(
                self._base_for_perspective, self._perspective_corners
            )

        # Зберігаємо знімок _base ДО будь-яких змін
        self._base_for_perspective = self._base.copy()
        base_snapshot = self._base.copy()
        corners_before = pipeline.detect_corners(base_snapshot, filename=self._current_path)

        # DEBUG: лог стану перед операцією
        self._logger.debug(
            "_do_persp_auto: base_snapshot shape=%s md5=%s",
            base_snapshot.shape,
            hashlib.md5(base_snapshot.tobytes()).hexdigest()[:16],
        )

        def _work():
            return pipeline.run_perspective_auto_smart(
                base_snapshot, self._settings, filename=self._current_path
            )

        def _on_done(payload):
            result, status = payload

            # DEBUG: лог результату
            self._logger.debug(
                "_do_persp_auto: result shape=%s md5=%s status=%s",
                result.shape,
                hashlib.md5(result.tobytes()).hexdigest()[:16],
                status,
            )

            # Визначаємо режим за статусом
            if "перспектива не" in status:
                # noop: нічого не знайдено
                self._base_for_perspective = None
                self._perspective_corners = None
                self._base = base_snapshot.copy()
                self._processed = result
                self._preview.set_before(image_utils.make_preview(self._base))
                self._preview.set_after(image_utils.make_preview(result))
                self._preview.disable_perspective_edit()
                self._clear_edit_mode()
                self._set_status(status)
            elif "deskew" in status and corners_before is None:
                # deskew only: НЕ комітимо в _base, показуємо тільки в "ПІСЛЯ"
                self._base_for_perspective = None
                self._perspective_corners = None
                # Зберігаємо результат як відкладений (користувач має підтвердити)
                self._pending_deskew_result = result.copy()
                self._set_edit_mode(EditMode.DESKEW_PENDING)
                self._processed = result
                # BEFORE — незмінний оригінал (base_snapshot)
                self._preview.set_before(image_utils.make_preview(base_snapshot))
                # AFTER — deskew-результат
                self._preview.set_after(image_utils.make_preview(result))
                self._preview.disable_perspective_edit()
                self._set_status(status + " | Натисніть «✔ Застосувати deskew» або «✖ Скасувати»")
                self._show_deskew_buttons(True)
            else:
                # straight/corrected: показати кути, увійти в perspective-режим
                corners = corners_before if corners_before is not None else pipeline.detect_corners(result)
                if corners is not None and self._base_for_perspective is not None:
                    # Завдання 3.3: валідація точок у межах зображення
                    if not self._validate_corners_in_bounds(corners, self._base_for_perspective):
                        self._logger.debug("_do_persp_auto: кути поза межами, fallback до default")
                        corners = self._default_perspective_corners(self._base_for_perspective)
                        status = "Документ не знайдено — встановіть точки вручну (auto-detect поза межами)"
                    self._perspective_corners = corners.copy()
                    self._show_perspective_points(corners, status)
                else:
                    self._perspective_corners = None
                    self._preview.disable_perspective_edit()
                    self._set_status(status)
                # Оновлюємо BEFORE з джерелом
                if self._base_for_perspective is not None:
                    prev_source = image_utils.make_preview(self._base_for_perspective)
                    # Завдання 3.5: фіксуємо розмір панелей (якщо увійшли в перспективу)
                    self._freeze_preview_panels()
                    self._preview.set_before(prev_source)
                # Результат (perspective applied) — показуємо на AFTER
                self._processed = result
                self._preview.set_after(image_utils.make_preview(result))

            self._on_controls_changed()
            self._update_buttons()

        self._run_in_background(_work, _on_done, button_to_lock=self._btn_autofix)

    def _commit_deskew(self) -> None:
        """Застосовує відкладений deskew-результат у _base."""
        if self._pending_deskew_result is None:
            return
        self._logger.debug("_commit_deskew: коміт deskew-результату в _base")
        result = self._pending_deskew_result.copy()
        self._pending_deskew_result = None
        self._commit_base_result(result, "Deskew застосовано", update_before=True)
        self._show_deskew_buttons(False)
        self._on_controls_changed()

    def _cancel_deskew(self) -> None:
        """Скасовує відкладений deskew-результат, повертається до base_snapshot."""
        if self._pending_deskew_result is None:
            return
        self._logger.debug("_cancel_deskew: скасування deskew")
        self._pending_deskew_result = None
        self._clear_edit_mode()
        # _processed повертаємо до _base (який не змінювався)
        if self._base is not None:
            self._processed = self._base.copy()
            self._preview.set_after(image_utils.make_preview(self._base))
        self._show_deskew_buttons(False)
        self._set_status("Deskew скасовано")

    def _show_deskew_buttons(self, visible: bool) -> None:
        """Показує/ховає кнопки «Застосувати deskew» та «Скасувати deskew»."""
        if hasattr(self, '_btn_apply_deskew') and self._btn_apply_deskew is not None:
            self._btn_apply_deskew.setVisible(visible)
        if hasattr(self, '_btn_cancel_deskew') and self._btn_cancel_deskew is not None:
            self._btn_cancel_deskew.setVisible(visible)

    def _corners_to_preview_pts(
        self,
        corners: np.ndarray,
        source: np.ndarray,
    ) -> list[QPoint]:
        """Масштабує кути з простору source у простір make_preview(source)."""
        prev = image_utils.make_preview(source)
        points = image_to_preview_points(
            corners, source.shape[:2], prev.shape[:2]
        )
        return [QPoint(int(point[0]), int(point[1])) for point in points]

    def _preview_pts_to_corners(
        self,
        points: list[QPoint],
        source: np.ndarray,
    ) -> np.ndarray:
        """Масштабує точки з простору make_preview(source) у простір source."""
        prev = image_utils.make_preview(source)
        values = [[point.x(), point.y()] for point in points]
        return preview_to_image_points(values, source.shape[:2], prev.shape[:2])

    def _freeze_preview_panels(self) -> None:
        """Фіксує розмір панелей прев'ю під час редагування перспективи (Завд. 3.5)."""
        for panel in (self._preview._before, self._preview._after):
            panel.setFixedSize(panel.size())

    def _unfreeze_preview_panels(self) -> None:
        """Відновлює автоматичний ресайз панелей прев'ю (Завд. 3.5)."""
        for panel in (self._preview._before, self._preview._after):
            panel.setMinimumSize(280, 280)
            panel.setMaximumSize(16777215, 16777215)
            panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _show_perspective_points(
        self, corners: np.ndarray, status_msg: str
    ) -> None:
        """corners — у просторі _base_for_perspective."""
        if self._base_for_perspective is None:
            return
        # Оновлюємо BEFORE: показуємо джерело (до перспективи)
        prev_source = image_utils.make_preview(self._base_for_perspective)
        self._preview.set_before(prev_source)
        # Масштабуємо кути
        pts = self._corners_to_preview_pts(corners, self._base_for_perspective)
        self._preview.enable_perspective_edit(pts)
        self._set_status(status_msg)

    def _validate_corners_in_bounds(self, corners: np.ndarray, image: np.ndarray) -> bool:
        """Перевіряє чи всі кути в межах [-20%, 120%] розміру зображення."""
        h, w = image.shape[:2]
        for c in corners:
            if c[0] < -w * 0.2 or c[0] > w * 1.2 or c[1] < -h * 0.2 or c[1] > h * 1.2:
                return False
        return True

    def _default_perspective_corners(self, image: np.ndarray) -> np.ndarray:
        """Повертає кути 80% центру зображення (10% відступ з кожного краю)."""
        h, w = image.shape[:2]
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        return np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y],
        ], dtype=np.float32)

    def _do_persp_manual_fallback(self):
        """Ручний режим з дефолтними точками (80% центру _base_for_perspective)."""
        source = self._base_for_perspective if self._base_for_perspective is not None else self._base
        if source is None:
            return
        corners = self._default_perspective_corners(source)
        self._perspective_corners = corners
        pts = self._corners_to_preview_pts(corners, source)
        self._preview.enable_perspective_edit(pts)
        self._set_status("Документ не знайдено — встановіть точки вручну")

    def _do_persp_manual(self) -> None:
        """Ручна корекція: знімок, детекція, показ точок.
        TODO1.5: коміт відбувається по mouseRelease, не при повторному натисканні кнопки."""
        if self._base is None:
            return
        self._set_edit_mode(EditMode.PERSPECTIVE)
        # Якщо вже в режимі ручної перспективи — просто перезапускаємо без коміту
        if self._base_for_perspective is not None and self._perspective_corners is not None:
            self._preview.disable_perspective_edit()
            self._perspective_cached_result = None
        else:
            # Зберігаємо знімок джерела
            self._base_for_perspective = self._base.copy()
            self._preview.disable_perspective_edit()
            self._perspective_cached_result = None
        # TODO1.5: скидаємо флаг реального перетягування при вході в режим
        self._persp_drag_applied = False

        # DEBUG: лог стану перед операцією
        self._logger.debug(
            "_do_persp_manual: base_for_perspective shape=%s md5=%s",
            self._base_for_perspective.shape,
            hashlib.md5(self._base_for_perspective.tobytes()).hexdigest()[:16],
        )
        # Спробуємо детектувати кути
        corners = pipeline.detect_corners(self._base_for_perspective, filename=self._current_path)
        if corners is None:
            # Спроба 2: з пониженим порогом площі через _try_external_contour
            from processing.perspective import _try_external_contour, _refine_corners_subpix, _order_points
            import cv2 as _cv2
            _gray = _cv2.cvtColor(self._base_for_perspective, _cv2.COLOR_BGR2GRAY)
            _small_scale = min(800 / max(_gray.shape[:2]), 1.0)
            _small = _cv2.resize(_gray, None, fx=_small_scale, fy=_small_scale) if _small_scale < 1.0 else _gray
            corners = _try_external_contour(_small)
            if corners is not None:
                corners = (corners / _small_scale).astype(np.float32)
                corners = _refine_corners_subpix(_gray, corners)
                status = "Тягніть кути для корекції перспективи (знайдено резервним методом)" + \
                    " | Перетягуйте кольорові точки (TL=червона, TR=зелена, BR=синя, BL=жовта)"
            else:
                corners = self._default_perspective_corners(self._base_for_perspective)
                status = "Встановіть кути вручну (документ не знайдено)"
        else:
            # Завдання 7.1: валідація — чи точки в межах [-20%, 120%] від розмірів зображення.
            # Використовуємо спільний метод _validate_corners_in_bounds (TODO 3.3-b),
            # щоб уникнути дублювання логіки з _do_persp_auto.
            if not self._validate_corners_in_bounds(corners, self._base_for_perspective):
                self._logger.debug("_do_persp_manual: точки за межами +-20%, fallback до default")
                corners = self._default_perspective_corners(self._base_for_perspective)
                status = "Документ не знайдено — встановіть точки вручну (auto-detect поза межами)"
            else:
                status = "Тягніть кути для корекції перспективи" + \
                    " | Перетягуйте кольорові точки (TL=червона, TR=зелена, BR=синя, BL=жовта)"
        # Зберігаємо кути у просторі зображення
        self._perspective_corners = corners
        # Оновлюємо BEFORE: показуємо джерело (до перспективи)
        prev_source = image_utils.make_preview(self._base_for_perspective)
        self._preview.set_before(prev_source)
        # Показуємо live результат на AFTER
        try:
            persp_result = pipeline.run_perspective_manual(
                self._base_for_perspective, corners
            )
            self._processed = persp_result
            self._preview.set_after(image_utils.make_preview(persp_result))
        except Exception:
            self._preview.set_after(prev_source)
        # Завдання 3.5: фіксуємо розмір панелей
        self._freeze_preview_panels()
        # Виставляємо точки на BEFORE
        pts = self._corners_to_preview_pts(corners, self._base_for_perspective)
        self._preview.enable_perspective_edit(pts)
        self._set_status(status)
        # Застосовуємо слайдери до результату перспективи
        self._on_controls_changed()
        self._update_buttons()

    def _do_persp_reset(self):
        """Скидає перспективу до стану до початку ручної перспективи."""
        if self._orig is None:
            return
        if self._base_for_perspective is not None:
            # Повертаємось до знімка, зробленого перед ручною перспективою
            # Це зберігає Auto Fix та інші корекції
            result = self._base_for_perspective.copy()
            self._base_for_perspective = None
        else:
            # Якщо ручна перспектива не починалась — скидаємо до оригіналу
            result = self._orig.copy()
        self._commit_base_result(result, "Перспективу скинуто", update_before=True)
        self._preview.disable_perspective_edit()
        # Завдання 3.5: розморожуємо панелі
        self._unfreeze_preview_panels()
        self._perspective_corners = None  # скидаємо збережені кути
        self._perspective_cached_result = None
        self._clear_edit_mode()
        # Після скидання перспективи застосовуємо поточні слайдери
        self._on_controls_changed()

    def _do_reset_all(self):
        """Скидає всі корекції до оригінального зображення."""
        if self._orig is None:
            return
        result = self._orig.copy()
        self._base_for_perspective = None
        self._commit_base_result(
            result,
            "Всі корекції скинуто",
            update_before=True,
            update_autofix=True,
        )
        self._perspective_corners = None  # скидаємо збережені кути перспективи
        self._perspective_cached_result = None
        self._clear_edit_mode()
        # Зберігаємо скинуті налаштування для поточного файлу
        self._store_current_settings()

    def _on_persp_pts_light(self, points: list) -> None:
        """Завдання 3.1: Легкий обробник — тільки перемальовка точок на BEFORE.
        Викликається при кожному mousemove (points_changed)."""
        if self._base_for_perspective is None or len(points) != 4:
            return
        # Тільки оновлюємо відображення точок (self._before.update() вже викликано)
        # Ніяких важких обчислень

    def _on_persp_pts_heavy(self, points: list) -> None:
        """Оновлює live-прев’ю ручної перспективи після відпускання точки.

        Сесію не завершуємо в `mouseReleaseEvent`: точки мають залишатися
        доступними для наступних корекцій, а `_base` змінюється лише коли
        сесія явно завершується зовнішньою дією.
        """
        if self._base_for_perspective is None or len(points) != 4:
            return
        try:
            corners = self._preview_pts_to_corners(points, self._base_for_perspective)
            self._perspective_corners = corners
            persp_result = pipeline.run_perspective_manual(
                self._base_for_perspective, corners
            )
            # Кешуємо результат перспективи для _on_controls_changed
            self._perspective_cached_result = persp_result
            self._persp_drag_applied = True
            self._processed = persp_result.copy()
            self._preview.set_after(image_utils.make_preview(persp_result))
            self._update_buttons()
        except Exception as e:
            self._logger.error(f"Помилка перспективи: {e}", exc_info=True)
            self._set_status(f"Помилка перспективи: {e}")

    # ------------------------------------------------------------------
    # TODO2 крок 3: дефолтна стартова рамка кадрування
    # ------------------------------------------------------------------

    def _invalidate_crop_default_cache(self) -> None:
        """TODO2 крок 3.3: скидає кеш дефолтної рамки кадрування.

        Викликається в усіх місцях, де змінюється self._base.
        """
        self._crop_default_corners_full = None
        self._crop_default_corners_base_id = None

    def _active_crop_source(self) -> Optional[np.ndarray]:
        """Повертає стабільне джерело координат поточної crop-сесії."""
        try:
            source = getattr(self, "_crop_source", None)
        except RuntimeError:
            source = None
        return source if source is not None else self._base

    def _ensure_crop_source(self) -> Optional[np.ndarray]:
        """Створює незмінне джерело при першому вході в crop-сесію.

        Панель «До» показує оригінал, тому crop-сесія також працює від
        оригіналу, а повний Auto Fix після commit запускається вже на
        вибраному фрагменті. Це не змішує координати «До» з `_base`.
        """
        source = self._active_crop_source()
        if source is None:
            return None
        if source is self._base:
            try:
                original = getattr(self, "_orig", None)
            except RuntimeError:
                original = None
            source = original.copy() if original is not None else source.copy()
            self._crop_source = source
        return source

    def _full_frame_crop_corners(self, image: np.ndarray) -> np.ndarray:
        """TODO2 крок 3.5: повертає 4 кути точно по межах зображення.

        На відміну від _default_perspective_corners (80% центр з відступом),
        тут потрібен "весь кадр без штучного відступу".
        """
        h, w = image.shape[:2]
        return np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

    def _cleanup_crop_detect_thread(self) -> None:
        """TODO2 крок 3.2: безпечно завершує crop-detect потік та звільняє ресурси."""
        if self._crop_detect_thread is not None:
            if self._crop_detect_thread.isRunning():
                self._crop_detect_thread.quit()
                if not self._crop_detect_thread.wait(3000):
                    self._logger.warning("CropDetectWorker thread did not finish within 3s timeout")
            self._crop_detect_thread.deleteLater()
            self._crop_detect_thread = None
        if self._crop_detect_worker is not None:
            self._crop_detect_worker.deleteLater()
            self._crop_detect_worker = None
        self._crop_detection_base_reference = None
        self._crop_detection_base_id = None
        self._crop_detection_started = 0.0

    def _on_crop_session_requested(self) -> None:
        """Показує безпечну стартову рамку по межах поточного зображення.

        Кадрування не залежить від фонової авто-детекції: користувач
        синхронно отримує активну рамку точно по межах зображення.
        """
        if self._base is None:
            return
        self._set_edit_mode(EditMode.CROP)
        source = self._ensure_crop_source()
        if source is None:
            return
        base_id = id(source)
        if (self._crop_default_corners_full is not None
                and self._crop_default_corners_base_id == base_id):
            self._preview._before.set_crop_rect(
                self._corners_to_preview_pts(self._crop_default_corners_full, source)
            )
            self._preview._before.set_crop_ready(True)
            return

        # TODO: авто-детекція контуру документа тимчасово вимкнена — рамка
        # стартує на весь кадр (рішення Блоку 3, 2026-08).
        corners_full = self._full_frame_crop_corners(source)
        self._crop_default_corners_full = corners_full.copy()
        self._crop_default_corners_base_id = base_id
        self._preview._before.set_crop_rect(
            self._corners_to_preview_pts(corners_full, source)
        )
        self._preview._before.set_crop_ready(True)

    def _on_crop_detection_finished(self, corners_full) -> None:
        base_reference = self._crop_detection_base_reference
        base_id = self._crop_detection_base_id
        started = self._crop_detection_started
        if base_reference is None or base_id is None:
            self._crop_detection_in_progress = False
            return
        self._logger.debug("crop detection: %.2f ms, input=%s", (time.perf_counter() - started) * 1000, base_reference.shape)
        if self._base is not base_reference or id(self._base) != base_id:
            self._crop_detection_in_progress = False
            return
        if corners_full is None:
            corners_full = self._full_frame_crop_corners(base_reference)
        self._crop_default_corners_full = np.asarray(corners_full, dtype=np.float32).copy()
        self._crop_default_corners_base_id = base_id
        self._preview._before.set_crop_rect(
            self._corners_to_preview_pts(self._crop_default_corners_full, base_reference)
        )
        self._crop_detection_in_progress = False

    def _on_crop_detection_error(self, message) -> None:
        base_reference = self._crop_detection_base_reference
        base_id = self._crop_detection_base_id
        self._logger.warning("crop detection failed: %s", message)
        if base_reference is not None and self._base is base_reference and id(self._base) == base_id:
            self._crop_default_corners_full = self._full_frame_crop_corners(base_reference)
            self._crop_default_corners_base_id = base_id
            self._preview._before.set_crop_rect(
                self._corners_to_preview_pts(self._crop_default_corners_full, base_reference)
            )
        self._crop_detection_in_progress = False

    def _on_crop_preview_changed(self, _points: list) -> None:
        """Оновлює AFTER під час hover-редагування, не змінюючи `_base`.

        Це дає користувачу негайне візуальне підтвердження crop/перспективи,
        тоді як остаточний коміт відбувається лише після виходу з hover-сесії.
        """
        source = self._active_crop_source()
        if source is None:
            return
        crop_pts_preview, persp_pts_preview, persp_detached = self._preview.get_crop_state()
        if len(crop_pts_preview) != 4 or len(persp_pts_preview) != 4:
            return
        try:
            crop_corners_full = self._preview_pts_to_corners(crop_pts_preview, source)
            cropped = pipeline.run_crop_rect(source, crop_corners_full)
            if all(not detached for detached in persp_detached):
                result = cropped
            else:
                persp_corners_full = self._preview_pts_to_corners(
                    persp_pts_preview, source
                )
                x_min = float(crop_corners_full[:, 0].min())
                y_min = float(crop_corners_full[:, 1].min())
                persp_corners_full[:, 0] -= x_min
                persp_corners_full[:, 1] -= y_min
                result = pipeline.run_crop_pin_perspective(cropped, persp_corners_full)
            self._processed = result.copy()
            self._preview.set_after(image_utils.make_preview(result))
        except Exception as exc:
            self._logger.error("Помилка live-прев’ю кадрування: %s", exc, exc_info=True)

    def _on_crop_session_committed(self) -> None:
        """Комітить crop, перспективу і, за потреби, Auto Fix одним Undo."""
        source = self._active_crop_source()
        if self._base is None or source is None:
            return
        self._set_edit_mode(EditMode.CROP)

        path_snapshot = self._current_path
        base_reference = self._base
        source_snapshot = source.copy()
        crop_pts_preview, persp_pts_preview, persp_detached = self._preview.get_crop_state()
        if len(crop_pts_preview) != 4 or len(persp_pts_preview) != 4:
            return

        try:
            crop_corners_full = self._preview_pts_to_corners(crop_pts_preview, source_snapshot)
            cropped = pipeline.run_crop_rect(source_snapshot, crop_corners_full)
            if all(not detached for detached in persp_detached):
                final = cropped
            else:
                persp_corners_full = self._preview_pts_to_corners(
                    persp_pts_preview, source_snapshot
                )
                x_min = float(crop_corners_full[:, 0].min())
                y_min = float(crop_corners_full[:, 1].min())
                persp_corners_cropped = persp_corners_full.copy()
                persp_corners_cropped[:, 0] -= x_min
                persp_corners_cropped[:, 1] -= y_min
                final = pipeline.run_crop_pin_perspective(
                    cropped, persp_corners_cropped
                )
        except Exception as exc:
            self._logger.error("Помилка коміту кадрування: %s", exc, exc_info=True)
            self._set_status(f"Помилка кадрування: {exc}")
            return

        if self._current_path != path_snapshot or self._base is not base_reference:
            self._logger.warning("Застарілий коміт кадрування відкинуто")
            return

        def _commit_result(result: np.ndarray, status_msg: str, autofix_applied: str | None = None):
            if self._current_path != path_snapshot or self._base is not base_reference:
                self._logger.warning("Застарілий коміт кадрування відкинуто")
                return
            self._commit_base_result(
                result,
                status_msg,
                autofix_applied,
                update_before=False,
                update_autofix=True,
            )
            self._base_for_perspective = None
            self._perspective_corners = None
            self._perspective_cached_result = None
            # «До» та crop-рамка лишаються у координатах crop_source.
            # Це дозволяє повторно змінювати вже вибрану область без
            # підміни зображення у панелі «До».
            self._clear_edit_mode()

        after_crop_action = self._settings.get("after_crop_action")
        if after_crop_action is None:
            after_crop_action = (
                "autofix" if self._settings.get("auto_apply_autofix", True)
                else "cropped_only"
            )
        if after_crop_action != "autofix":
            _commit_result(final, "Кадрування застосовано")
            return

        vals = self._controls.values()

        def _work():
            return self._compute_full_autofix(final.copy(), vals)

        def _on_done(payload):
            result, status_msg, log_entries = payload
            if self._current_path != path_snapshot or self._base is not base_reference:
                self._logger.debug("Кадрування: файл змінився, ігноруємо застарілий результат")
                return
            if log_entries:
                applied = [entry for entry in log_entries if entry.get("applied")]
                type_parts = [entry["detail"] for entry in applied
                              if entry["step"] in ("doc_type", "color_mode")]
                other_parts = [entry["detail"] for entry in applied
                               if entry["step"] not in ("doc_type", "color_mode")]
                detailed_status = "Auto Fix: " + " | ".join(type_parts + other_parts)
            else:
                detailed_status = status_msg
            _commit_result(result, detailed_status, "auto_fix")

        self._run_in_background(_work, _on_done, button_to_lock=self._btn_autofix)

# ------------------------------------------------------------------
# Друк та збереження
# ------------------------------------------------------------------

    def _do_print_current(self):
        # Якщо є незафіксований deskew — фіксуємо
        if self._pending_deskew_result is not None:
            self._commit_deskew()
        # Якщо є незафіксована ручна перспектива — фіксуємо (TODO1.5)
        self._commit_pending_perspective()
        # Правильна перевірка numpy array через "is not None"
        image = self._processed if self._processed is not None else self._base
        if image is None:
            self._set_status("Немає зображення для друку")
            return
        try:
            # Зберігаємо налаштування перед друком
            self._store_current_settings()
            # TODO 2.5-b: синхронізуємо нові файли з GUI-черги без скидання
            # позиції друку (_index). Безумовно, щоб файли, додані після
            # часткового проходу черги, потрапили в _processor і не були
            # "загублені" при подальшій навігації через _load_next_manual.
            self._processor.sync_new_files(self._queue.get_all_paths())
            # Якщо файл відкритий вручну, використовуємо _current_path
            if self._current_path and self._current_path in self._queue.get_all_paths():
                # Знаходимо індекс файлу в черзі
                all_paths = self._queue.get_all_paths()
                idx = all_paths.index(self._current_path)
                # Друкуємо напряму через printer_module
                s = self._settings
                from core import saver, printer as printer_module
                # Зберігаємо тільки якщо налаштована save_folder (ніколи не перезаписуємо оригінал)
                save_folder = s.get("save_folder", "")
                if save_folder:
                    output_path = file_utils.build_output_path(self._current_path, save_folder, suffix="_edited")
                    saver.save(image, output_path, quality=s.get("jpg_quality", 95))
                printer_module.print_image(
                    image,
                    printer_name=s.get("printer_name", ""),
                    jpg_quality=s.get("jpg_quality", 95),
                )
                self._queue.mark_done(idx)
                self._set_status(f"Надруковано: {os.path.basename(self._current_path)}")
            else:
                # Батч режим - використовуємо processor
                printed_path = self._processor.print_current(image)
                idx = self._processor.current_index - 1
                self._queue.mark_done(idx)
                self._set_status(f"Надруковано: {os.path.basename(printed_path)}")
            self._load_next_manual()
        except Exception as e:
            self._logger.error(f"Помилка друку: {e}", exc_info=True)
            self._set_status(f"Помилка друку: {e}")

    def _do_skip(self):
        skipped = self._processor.skip_current()
        idx = self._processor.current_index - 1
        if idx >= 0:
            self._queue.mark_skipped(idx)
        self._set_status(
            f"Пропущено: {os.path.basename(skipped)}" if skipped else "Пропущено"
        )
        self._load_next_manual()

    def _do_save_image(self):
        """Зберігає поточне зображення — для відладки."""
        image = self._processed if self._processed is not None else self._base
        if image is None:
            self._set_status("Немає зображення для збереження")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти зображення", "result.jpg", "JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            from core import saver
            saver.save(image, path, quality=self._settings.get("jpg_quality", 95))
            self._set_status(f"Збережено: {os.path.basename(path)}")
        except Exception as e:
            self._logger.error(f"Помилка збереження: {e}", exc_info=True)
            self._set_status(f"Помилка збереження: {e}")

    def _do_print_all(self):
        all_paths = self._queue.get_all_paths()
        if not all_paths:
            self._set_status("Черга порожня")
            return
        self._processor.set_files(all_paths)
        if self._radio_auto.isChecked():
            self._start_auto()
        else:
            self._load_next_manual()

    # ------------------------------------------------------------------
    # Авто-режим у потоці
    # ------------------------------------------------------------------

    def _cleanup_auto_thread(self):
        """Безпечно завершує авто-потік та звільняє ресурси."""
        if hasattr(self, '_auto_thread') and self._auto_thread is not None:
            if self._auto_thread.isRunning():
                self._auto_thread.quit()
                if not self._auto_thread.wait(5000):
                    self._logger.warning("AutoWorkerThread did not finish within 5s timeout")
            self._auto_thread.deleteLater()
            self._auto_thread = None
        if hasattr(self, '_worker') and self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _start_auto(self):
        # Якщо попередній авто-потік ще виконується — чекаємо його завершення
        if hasattr(self, '_auto_thread') and self._auto_thread is not None and self._auto_thread.isRunning():
            self._logger.warning("Попередній AutoWorkerThread ще виконується — чекаємо завершення")
            self._cleanup_auto_thread()

        self._progress.setVisible(True)
        self._progress.setRange(0, self._processor.total)
        self._set_buttons_enabled(False)
        self._radio_auto.setEnabled(False)
        self._radio_manual.setEnabled(False)

        self._auto_thread = QThread()
        self._auto_thread.setObjectName("AutoWorkerThread")
        self._worker = AutoWorker(self._processor)
        self._worker.moveToThread(self._auto_thread)
        self._auto_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_auto_progress)
        self._worker.print_progress.connect(self._on_auto_print_progress)
        self._worker.error.connect(self._on_auto_error)
        self._worker.finished.connect(self._on_auto_done)
        self._worker.finished.connect(self._auto_thread.quit)
        self._auto_thread.finished.connect(self._auto_thread.deleteLater)
        self._auto_thread.start()

    def _on_auto_progress(self, cur: int, total: int, fname: str):
        """Завдання 2.4: прогрес фази 1 (обробка)."""
        self._progress.setValue(cur)
        self._set_status(f"Обробка {cur}/{total}: {fname}")

    def _on_auto_print_progress(self, cur: int, total: int, fname: str):
        """Завдання 2.4: прогрес фази 2 (друк)."""
        self._progress.setValue(total + cur)  # total = offset після фази 1
        self._set_status(f"Друк {cur}/{total}: {fname}")

    def _on_auto_error(self, idx: int, fname: str, msg: str):
        self._queue.mark_error(idx)
        self._set_status(f"Помилка [{idx+1}]: {fname} — {msg}")

    def _on_auto_done(self, count: int):
        self._progress.setVisible(False)
        self._set_buttons_enabled(True)
        self._radio_auto.setEnabled(True)
        self._radio_manual.setEnabled(True)
        self._set_status(f"Готово. Надруковано {count} з {self._processor.total}")
        # Позначаємо всі що не мають статусу (перевірка через get_status,
        # а не через текст — TODO 4.5)
        for i in range(self._queue.count()):
            if self._queue.get_status(i) not in ("done", "error", "skipped"):
                self._queue.mark_done(i)

    # ------------------------------------------------------------------
    # Ручний режим — крокування по черзі
    # ------------------------------------------------------------------

    def _load_next_manual(self):
        # Зберігаємо налаштування поточного файлу перед переходом
        self._store_current_settings()
        # Commit deskew та перспективи перед переходом (TODO1.5)
        if self._pending_deskew_result is not None:
            self._commit_deskew()
        self._commit_pending_perspective()
        if not self._processor.has_next():
            self._set_status("Всі файли оброблено ✓")
            self._preview.clear()
            self._orig = None
            self._base = None
            self._processed = None
            self._current_path = None
            self._update_buttons()
            return
        try:
            idx = self._processor.current_index
            self._queue.mark_current(idx)
            img = self._processor.load_current()
            self._orig = img
            self._base = img.copy()  # скидаємо базове зображення
            self._invalidate_crop_default_cache()  # TODO2 крок 3.3
            self._processed = None
            path = self._processor.current_file()
            self._current_path = path
            if path is not None:
                self._restore_file_settings(path)
            prev = image_utils.make_preview(img)
            self._preview.set_before(prev)
            self._preview.set_after(prev)
            total = self._processor.total
            self._set_status(
                f"[{idx + 1}/{total}]  {os.path.basename(path or '')}"
            )
            self._update_buttons()
        except Exception as e:
            self._logger.error(f"Помилка завантаження з черги: {e}", exc_info=True)
            self._set_status(f"Помилка завантаження: {e}")

    # ------------------------------------------------------------------
    # Фонова обробка (SingleImageWorker)
    # ------------------------------------------------------------------

    def _cleanup_single_thread(self):
        """Безпечно завершує single-image потік та звільняє ресурси."""
        if self._single_thread is not None:
            if self._single_thread.isRunning():
                self._single_thread.quit()
                if not self._single_thread.wait(3000):
                    self._logger.warning("SingleImageWorker thread did not finish within 3s timeout")
            self._single_thread.deleteLater()
            self._single_thread = None
        if self._single_worker is not None:
            self._single_worker.deleteLater()
            self._single_worker = None

    def _reset_progress_ui(self, button_to_lock=None):
        """Скидає UI прогрес-бару та кнопок після завершення фонової операції."""
        self._progress.setVisible(False)
        self._progress.setRange(0, 100)
        self._set_buttons_enabled(True)
        if button_to_lock:
            button_to_lock.setEnabled(True)
            original_text = button_to_lock.property("_orig_text")
            if original_text is not None:
                button_to_lock.setText(original_text)
                button_to_lock.setProperty("_orig_text", None)

    def _run_in_background(self, func, on_finished, button_to_lock=None):
        """
        Запускає func() у фоновому потоці.
        on_finished(result) викликається в GUI-потоці після завершення.
        button_to_lock — кнопка яку заблокувати під час виконання.
        """
        # Якщо попередній потік ще живий — чекаємо його завершення
        if self._single_thread is not None and self._single_thread.isRunning():
            self._logger.warning("Попередній потік ще виконується — чекаємо завершення")
            self._cleanup_single_thread()

        # Показати прогрес
        self._progress.setRange(0, 0)   # indeterminate (пульсуючий)
        self._progress.setVisible(True)
        if button_to_lock:
            button_to_lock.setProperty("_orig_text", button_to_lock.text())
            button_to_lock.setEnabled(False)
            button_to_lock.setText("⏳ Обробка…")
        self._set_buttons_enabled(False)

        self._single_thread = QThread()
        self._single_thread.setObjectName("SingleImageWorkerThread")
        self._single_worker = SingleImageWorker(func)
        self._single_worker.moveToThread(self._single_thread)

        self._single_thread.started.connect(self._single_worker.run)

        def _on_done(result):
            self._reset_progress_ui(button_to_lock)
            self._cleanup_single_thread()
            on_finished(result)

        def _on_error(msg):
            self._reset_progress_ui(button_to_lock)
            self._cleanup_single_thread()
            self._set_status(f"Помилка: {msg}")
            self._logger.error(msg)

        self._single_worker.finished.connect(_on_done)
        self._single_worker.error.connect(_on_error)
        self._single_thread.finished.connect(self._single_thread.deleteLater)
        self._single_thread.start()

    # ------------------------------------------------------------------
    # Допоміжне
    # ------------------------------------------------------------------

    def _update_buttons(self):
        has_queue = bool(self._queue.get_all_paths())
        has_img   = self._orig is not None
        is_batch  = self._radio_auto.isChecked()
        try:
            session = getattr(self, "_edit_session", None)
        except RuntimeError:
            session = None
        in_persp  = session is not None and session.mode is EditMode.PERSPECTIVE
        busy      = self._has_running_threads()

        self._btn_print_all.setEnabled(has_queue and is_batch and not busy)
        self._btn_print_all.setToolTip(
            "" if is_batch else "Доступно тільки в Пакетному режимі"
        )
        self._btn_print.setEnabled(has_img and not busy)
        self._btn_skip.setEnabled(
            self._processor.has_next() and not is_batch and not busy
        )
        # Авто Фікс доступний і під час редагування перспективи
        self._btn_autofix.setEnabled(has_img and not busy)
        self._btn_shadow_remove.setEnabled(has_img and not busy)
        self._btn_universal.setEnabled(has_img and not busy)
        self._btn_save_img.setEnabled(has_img and not busy)
        # Undo/Redo — тільки ручний режим, неактивні під час фонових операцій
        self._btn_undo.setEnabled(has_img and not busy)
        self._btn_redo.setEnabled(has_img and not busy)
        # Кнопка скидання перспективи — тільки в perspective-режимі
        # (вона вже є в controls, не потребує додаткового управління)

    def _set_buttons_enabled(self, enabled: bool):
        for b in (self._btn_autofix, self._btn_shadow_remove, self._btn_universal, self._btn_print, self._btn_skip, self._btn_print_all,
                  self._btn_undo, self._btn_redo):
            b.setEnabled(enabled)

    def _set_status(self, text: str, timeout_ms: int = 0):
        sb = self.statusBar()
        if sb is None:
            return
        if text.startswith("Auto Fix:"):
            sb.setStyleSheet(_STATUS_BAR_STYLE_SUCCESS)
        elif "Помилка" in text:
            sb.setStyleSheet(_STATUS_BAR_STYLE_ERROR)
        else:
            sb.setStyleSheet(_STATUS_BAR_STYLE)
        sb.showMessage(text, timeout_ms)

    def _reset_status_style(self):
        """Скидає стиль статус-бару до стандартного."""
        sb = self.statusBar()
        if sb is not None:
            sb.setStyleSheet(_STATUS_BAR_STYLE)

    def _set_file_status(self, filename: str):
        self._status_file_label.setText(filename)

    # ------------------------------------------------------------------
    # Debug dump для GUI-тестувальника
    # ------------------------------------------------------------------

    def _dump_widget_geometries(self) -> None:
        """Записує геометрію всіх віджетів з objectName у JSON для GUITester."""
        import json
        from PyQt6.QtCore import QRect

        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "widgets_debug.json"

        def _rect_to_dict(r: QRect) -> dict:
            return {"x": r.x(), "y": r.y(), "width": r.width(), "height": r.height()}

        widgets = {}

        def _collect(widget, depth: int = 0) -> None:
            name = widget.objectName()
            if name:
                gr = widget.geometry()
                # Convert to global coordinates
                global_pos = widget.mapToGlobal(widget.rect().topLeft())
                widgets[name] = {
                    "x": global_pos.x(),
                    "y": global_pos.y(),
                    "width": gr.width(),
                    "height": gr.height(),
                    "class": type(widget).__name__,
                }
            for child in widget.findChildren(QWidget):
                child_name = child.objectName()
                if child_name and child_name not in widgets:
                    global_pos = child.mapToGlobal(child.rect().topLeft())
                    child_geom = child.geometry()
                    widgets[child_name] = {
                        "x": global_pos.x(),
                        "y": global_pos.y(),
                        "width": child_geom.width(),
                        "height": child_geom.height(),
                        "class": type(child).__name__,
                    }

        _collect(self)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(widgets, f, indent=2, ensure_ascii=False)

        self._logger.debug(f"Widget geometries dumped to {out_path} ({len(widgets)} widgets)")
