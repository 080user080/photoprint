"""
Головне вікно програми PhotoPrint.
Drag & Drop через WM_DROPFILES (utils/win_drop.py) — перевірено на Windows 10/11.

ВИПРАВЛЕННЯ: Зберігаємо базове зображення після перспективної корекції,
щоб слайдери працювали з виправленою перспективою.
"""

import os
import sys
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QButtonGroup, QRadioButton,
    QFileDialog, QProgressBar, QScrollArea, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QPoint

# WM_DROPFILES — єдиний надійний механізм Drag&Drop на Windows 10/11 з PyQt6
if sys.platform == "win32":
    from utils.win_drop import register_drop_window, DropEventFilter

from gui.preview         import PreviewPanel
from gui.queue_view      import QueueView
from gui.controls        import ControlsPanel
from gui.settings_window import SettingsWindow
from batch.batch_processor import BatchProcessor
from processing import pipeline
from utils import file_utils, image_utils
from utils.logger import get_logger
from config import app_settings


# ---------------------------------------------------------------------------
# Worker для авто-режиму (окремий потік — GUI не зависає)
# ---------------------------------------------------------------------------

class AutoWorker(QObject):
    progress = pyqtSignal(int, int, str)   # (1-based index, total, filename)
    error    = pyqtSignal(int, str, str)   # (index, filename, message)
    finished = pyqtSignal(int)             # кількість надрукованих

    def __init__(self, processor: BatchProcessor):
        super().__init__()
        self._p = processor

    def run(self):
        count = self._p.run_auto(
            on_progress=lambda c, t, f: self.progress.emit(c, t, f),
            on_error=lambda i, f, m: self.error.emit(i, f, m),
        )
        self.finished.emit(count)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoPrint")
        self.setMinimumSize(1100, 680)

        self._logger = get_logger(__name__)
        self._settings   = app_settings.load()
        self._processor  = BatchProcessor(self._settings)
        self._orig:      np.ndarray | None = None
        self._base:      np.ndarray | None = None  # базове зображення після перспективи
        self._processed: np.ndarray | None = None
        self._auto_thread = None
        self._drop_filter = None
        self._current_path: str | None = None          # поточний файл у ручному/перегляді
        self._per_file: dict[str, dict] = {}            # збережені налаштування слайдерів по файлу

        self._settings_win = SettingsWindow()
        self._settings_win.settings_saved.connect(self._on_settings_saved)

        self._build_ui()
        self._apply_default_mode()
        self._update_buttons()

        # Drag & Drop реєструємо після показу вікна
        if sys.platform == "win32":
            QTimer.singleShot(300, self._setup_win_drop)

    def _setup_win_drop(self):
        hwnd = int(self.winId())
        register_drop_window(hwnd)
        self._drop_filter = DropEventFilter(self._on_win_drop)
        QApplication.instance().installNativeEventFilter(self._drop_filter)

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

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # === Ліва колонка: черга ===
        left = QVBoxLayout()
        left.setSpacing(4)

        lbl_q = QLabel("Черга файлів")
        lbl_q.setStyleSheet("font-weight:bold; color:#111111; font-size:13px;")

        self._queue = QueueView()
        # QueueView.files_dropped — резервний (якщо Qt DnD раптом спрацює)
        self._queue.files_dropped.connect(self._on_win_drop)
        self._queue.selection_changed.connect(self._on_queue_selection)

        btn_add    = QPushButton("Додати файли…")
        btn_folder = QPushButton("Додати папку…")
        btn_clear  = QPushButton("Очистити чергу")
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

        # === Центр: прев'ю ===
        center = QVBoxLayout()
        center.setSpacing(6)

        self._preview = PreviewPanel()
        self._preview.perspective_points_changed.connect(self._on_persp_pts)

        self._progress = QProgressBar()
        self._progress.setVisible(False)

        self._status = QLabel("Перетягніть файли або натисніть «Додати файли»")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color:#444444; font-size:12px;")
        self._status.setWordWrap(True)

        center.addWidget(self._preview, 1)
        center.addWidget(self._progress)
        center.addWidget(self._status)

        # === Права колонка: керування ===
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(320)
        right_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(8)
        right.setContentsMargins(4, 4, 4, 4)

        # Режим
        lbl_mode = QLabel("Режим обробки")
        lbl_mode.setStyleSheet("font-weight:bold; color:#111111; font-size:13px;")
        self._radio_auto   = QRadioButton("Авто")
        self._radio_manual = QRadioButton("Ручний")
        self._radio_auto.setStyleSheet("color:#111111;")
        self._radio_manual.setStyleSheet("color:#111111;")
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self._radio_auto,   0)
        self._mode_group.addButton(self._radio_manual, 1)

        # Кнопки дій
        self._btn_autofix   = QPushButton("⚡ Auto Fix")
        self._btn_print     = QPushButton("🖨  Друк")
        self._btn_skip      = QPushButton("⏭  Пропустити")
        self._btn_print_all = QPushButton("🖨  Друкувати все")
        self._btn_save_img  = QPushButton("💾  Зберегти зображення")

        for b in (self._btn_autofix, self._btn_print,
                  self._btn_skip, self._btn_print_all):
            b.setFixedHeight(34)
            b.setStyleSheet(self._btn_style())

        self._btn_save_img.setFixedHeight(30)
        self._btn_save_img.setStyleSheet(
            "background:#5A8A5A; color:white; border:none;"
            "border-radius:4px; padding:4px 8px; font-size:12px;"
        )

        self._btn_autofix.clicked.connect(self._do_autofix)
        self._btn_print.clicked.connect(self._do_print_current)
        self._btn_skip.clicked.connect(self._do_skip)
        self._btn_print_all.clicked.connect(self._do_print_all)
        self._btn_save_img.clicked.connect(self._do_save_image)

        # Слайдери
        self._controls = ControlsPanel()
        self._controls.changed.connect(self._on_controls_changed)
        self._controls.auto_brightness_clicked.connect(self._do_auto_brightness)
        self._controls.auto_contrast_clicked.connect(self._do_auto_contrast)
        self._controls.auto_sharpen_clicked.connect(self._do_auto_sharpen)
        self._controls.perspective_auto_clicked.connect(self._do_persp_auto)
        self._controls.perspective_manual_clicked.connect(self._do_persp_manual)

        # Налаштування
        btn_settings = QPushButton("⚙  Налаштування")
        btn_settings.setStyleSheet(self._btn_style("#555555"))
        btn_settings.clicked.connect(self._open_settings)

        right.addWidget(lbl_mode)
        right.addWidget(self._radio_auto)
        right.addWidget(self._radio_manual)
        right.addSpacing(4)
        right.addWidget(self._btn_autofix)
        right.addWidget(self._btn_print)
        right.addWidget(self._btn_skip)
        right.addWidget(self._btn_print_all)
        right.addWidget(self._btn_save_img)
        right.addSpacing(4)
        right.addWidget(self._controls)
        right.addStretch()
        right.addWidget(btn_settings)

        right_scroll.setWidget(right_widget)

        root.addLayout(left,   0)
        root.addLayout(center, 1)
        root.addWidget(right_scroll, 0)

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

    def _apply_default_mode(self):
        if self._settings.get("default_mode", "auto") == "auto":
            self._radio_auto.setChecked(True)
        else:
            self._radio_manual.setChecked(True)
        self._controls.set_shadow_highlight(self._settings.get("shadow_highlight_strength", 0.0))
        self._controls.set_sharpen(self._settings.get("sharpen_strength", 0.4))
        self._controls.set_hdr(self._settings.get("hdr_strength", 0.0))

    def _on_settings_saved(self, s: dict):
        self._settings = s
        self._processor = BatchProcessor(self._settings)
        self._processor.set_files(self._queue.get_all_paths())
        self._apply_default_mode()

    def _open_settings(self):
        self._settings_win.load_from_file()
        self._settings_win.show()
        self._settings_win.raise_()

    # ------------------------------------------------------------------
    # Черга
    # ------------------------------------------------------------------

    def _on_files_added(self, paths: list[str]):
        """Спільний обробник після додавання файлів будь-яким способом."""
        supported = file_utils.filter_supported(paths)
        if not supported:
            self._set_status("Жоден з файлів не підтримується")
            return
        was_empty = self._processor.total == 0
        self._processor.add_files(supported)
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
        self._processed = None
        self._current_path = None
        self._per_file.clear()
        self._update_buttons()
        self._set_status("Черга очищена")

    def _store_current_settings(self):
        """Зберігає поточні значення слайдерів для _current_path."""
        path = self._current_path
        if path is None:
            return
        self._per_file[path] = self._controls.values()

    def _restore_file_settings(self, path: str):
        """Відновлює слайдери для файлу path, або скидає до дефолту."""
        vals = self._per_file.get(path)
        if vals is None:
            self._controls.reset_all()
            return
        self._controls.set_shadow_highlight(vals.get("shadow_highlight", 0.0), silent=True)
        self._controls.set_brightness(vals.get("brightness", 0.0), silent=True)
        self._controls.set_contrast(vals.get("contrast", 0.0), silent=True)
        self._controls.set_sharpen(vals.get("sharpen_strength", 0.0), silent=True)
        self._controls.set_hdr(vals.get("hdr_strength", 0.0), silent=True)
        self._controls.set_grayscale(vals.get("grayscale", False), silent=True)
        # Не викликаємо _on_controls_changed тут - дозволяємо слайдерам працювати самостійно

    def _on_queue_selection(self, path: str):
        """Клік на файл у списку — завантажуємо для перегляду."""
        try:
            self._store_current_settings()
            from core import loader
            img = loader.load(path)
            self._orig = img
            self._base = img.copy()  # початково base = orig
            self._processed = None
            self._current_path = path
            self._restore_file_settings(path)
            prev = image_utils.make_preview(img)
            self._preview.set_before(prev)
            # Авто-застосування Auto Fix при завантаженні
            if self._settings.get("auto_apply_autofix", True):
                self._do_autofix()
            else:
                self._update_buttons()
        except Exception as e:
            self._logger.error(f"Помилка завантаження файлу {path}: {e}", exc_info=True)
            self._set_status(f"Помилка завантаження: {e}")

    # ------------------------------------------------------------------
    # Обробка зображень
    # ------------------------------------------------------------------

    def _do_autofix(self):
        if self._orig is None:
            self._set_status("Спочатку оберіть файл")
            return
        try:
            s = self._settings
            vals = self._controls.values()
            if s.get("autofix_enabled", True):
                result, status_msg = pipeline.run_autofix(
                    self._orig,
                    sharpen_strength=vals["sharpen_strength"],
                    hdr_strength=vals["hdr_strength"],
                    use_hdr=s.get("hdr_in_autofix", True),
                    use_perspective=s.get("auto_perspective", True),
                    bw_binary=s.get("bw_binary", False),
                    classify_bw_std_thresh=s.get("classify_bw_std_thresh", 20.0),
                    classify_edge_ratio_min=s.get("classify_edge_ratio_min", 0.03),
                    classify_line_count_min=s.get("classify_line_count_min", 3),
                    shadow_highlight_strength=sh_strength,
                )
                # Оновлюємо базове зображення після автофіксу
                self._base = result.copy()
                self._set_status(status_msg)
                self._preview.set_autofix_applied(True)
            else:
                # autofix_enabled=False: тільки ручні налаштування
                result = pipeline.run_manual_adjustments(
                    self._base,  # використовуємо базове зображення
                    brightness=vals["brightness"],
                    contrast=vals["contrast"],
                    sharpen_strength=vals["sharpen_strength"],
                    hdr_strength=vals["hdr_strength"],
                    grayscale=vals["grayscale"],
                    shadow_highlight_strength=vals["shadow_highlight"],
                )
                self._set_status("Ручні налаштування")
                self._preview.set_autofix_applied(False)
            if s.get("autofix_enabled", True) and vals["grayscale"]:
                result = pipeline.run_grayscale(result)
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            self._update_buttons()
        except Exception as e:
            self._logger.error(f"Помилка Auto Fix: {e}", exc_info=True)
            self._set_status(f"Помилка Auto Fix: {e}")

    def _on_controls_changed(self, vals: dict):
        """Миттєво оновлює прев'ю при зміні будь-якого слайдера."""
        if self._base is None:
            return
        try:
            # Зберігаємо поточні значення для активного файлу
            self._store_current_settings()
            # Використовуємо базове зображення (з перспективою якщо була)
            result = pipeline.run_manual_adjustments(
                self._base,
                brightness=vals["brightness"],
                contrast=vals["contrast"],
                sharpen_strength=vals["sharpen_strength"],
                hdr_strength=vals["hdr_strength"],
                grayscale=vals["grayscale"],
                shadow_highlight_strength=vals["shadow_highlight"],
            )
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            # Скидаємо індикатор Auto Fix при ручних налаштуваннях
            self._preview.set_autofix_applied(False)
        except Exception as e:
            self._logger.error(f"Помилка обробки слайдерів: {e}", exc_info=True)
            self._set_status(f"Помилка обробки: {e}")

    def _do_auto_brightness(self):
        if self._base is None:
            return
        s = self._settings
        result = pipeline.run_auto_brightness(
            self._base,
            percentile_low=s.get("auto_percentile_low", 5.0),
            percentile_high=s.get("auto_percentile_high", 95.0),
        )
        # Оновлюємо базове зображення після авто-яскравості
        self._base = result.copy()
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
        self._set_status("Авто-яскравість застосована")

    def _do_auto_contrast(self):
        if self._base is None:
            return
        s = self._settings
        result = pipeline.run_auto_contrast(
            self._base,
            percentile_low=s.get("auto_percentile_low", 5.0),
            percentile_high=s.get("auto_percentile_high", 95.0),
        )
        # Оновлюємо базове зображення після авто-контрасту
        self._base = result.copy()
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
        self._set_status("Авто-контраст застосований")

    def _do_auto_sharpen(self):
        if self._base is None:
            return
        s = self._settings
        result, strength = pipeline.run_auto_sharpen(
            self._base,
            threshold=s.get("autosharp_threshold", 80.0),
            max_strength=s.get("autosharp_max_strength", 0.7),
        )
        # Оновлюємо базове зображення після авто-різкості
        self._base = result.copy()
        self._processed = result
        self._controls.set_sharpen(strength)
        self._preview.set_after(image_utils.make_preview(result))
        if strength > 0:
            self._set_status(f"Авто-різкість застосована ({strength:.2f})")
        else:
            self._set_status("Зображення достатньо різке — різкість не потрібна")

    def _do_persp_auto(self):
        """Авто-детекція перспективи з fallback до ручного режиму."""
        if self._orig is None:
            return
        corners = pipeline.detect_corners(self._orig)
        if corners is not None:
            # Авто знайшло документ — застосовуємо + показуємо точки для підправлення
            result, _ = pipeline.run_perspective_auto(self._orig)
            self._base = result.copy()
            self._processed = result
            self._preview.set_before(image_utils.make_preview(result))
            self._preview.set_after(image_utils.make_preview(result))
            self._show_perspective_points(corners, "Перспективу виправлено — підправте точки якщо потрібно")
        else:
            # Fallback: документ не знайдено → ручний режим з дефолтними точками
            self._do_persp_manual_fallback()
        self._update_buttons()

    def _show_perspective_points(self, corners: np.ndarray, status_msg: str):
        """Показує 4 точки перспективи на прев'ю."""
        orig_h, orig_w = self._orig.shape[:2]
        prev = image_utils.make_preview(self._orig)
        prev_h, prev_w = prev.shape[:2]
        sx = prev_w / max(orig_w, 1)
        sy = prev_h / max(orig_h, 1)
        pts = [QPoint(int(p[0] * sx), int(p[1] * sy)) for p in corners]
        self._preview.enable_perspective_edit(pts)
        self._set_status(status_msg)

    def _do_persp_manual_fallback(self):
        """Ручний режим з дефолтними точками по кутах прев'ю."""
        if self._orig is None:
            return
        orig_h, orig_w = self._orig.shape[:2]
        prev = image_utils.make_preview(self._orig)
        prev_h, prev_w = prev.shape[:2]
        m = 2
        pts = [
            QPoint(m,          m),
            QPoint(prev_w - m, m),
            QPoint(prev_w - m, prev_h - m),
            QPoint(m,          prev_h - m),
        ]
        self._preview.enable_perspective_edit(pts)
        self._set_status("Документ не знайдено — встановіть точки вручну")

    def _do_persp_manual(self):
        """Ручна корекція: спочатку пробуємо знайти точки авто, інакше дефолтні."""
        if self._orig is None:
            return
        corners = pipeline.detect_corners(self._orig)
        if corners is not None:
            self._show_perspective_points(corners, "Тягніть точки для корекції перспективи")
        else:
            self._do_persp_manual_fallback()

    def _on_persp_pts(self, points: list):
        if self._orig is None or len(points) != 4:
            return
        try:
            orig_h, orig_w = self._orig.shape[:2]
            prev = image_utils.make_preview(self._orig)
            prev_h, prev_w = prev.shape[:2]
            # Координати точок — у системі прев'ю (≤900px).
            # Масштабуємо назад в оригінальний розмір.
            scale_x = orig_w / max(prev_w, 1)
            scale_y = orig_h / max(prev_h, 1)
            corners = np.array(
                [[p.x() * scale_x, p.y() * scale_y] for p in points],
                dtype=np.float32
            )
            result = pipeline.run_perspective_manual(self._orig, corners)
            # Оновлюємо базове зображення після ручної перспективи
            self._base = result.copy()
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
        except Exception as e:
            self._logger.error(f"Помилка перспективи: {e}", exc_info=True)
            self._set_status(f"Помилка перспективи: {e}")

# ------------------------------------------------------------------
# Друк та збереження
# ------------------------------------------------------------------

    def _do_print_current(self):
        # Правильна перевірка numpy array через "is not None"
        image = self._processed if self._processed is not None else self._base
        if image is None:
            self._set_status("Немає зображення для друку")
            return
        try:
            # Зберігаємо налаштування перед друком
            self._store_current_settings()
            # Синхронізуємо черги якщо ще не зроблено
            if self._processor.total == 0:
                self._processor.set_files(self._queue.get_all_paths())
            # Якщо файл відкритий вручну, використовуємо _current_path
            if self._current_path and self._current_path in self._queue.get_all_paths():
                # Знаходимо індекс файлу в черзі
                all_paths = self._queue.get_all_paths()
                idx = all_paths.index(self._current_path)
                # Друкуємо напряму через printer_module
                s = self._settings
                from core import saver, printer as printer_module
                if s.get("save_before_print", True):
                    saved_path = saver.save(image, self._current_path, quality=s.get("jpg_quality", 95))
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

    def _start_auto(self):
        self._progress.setVisible(True)
        self._progress.setRange(0, self._processor.total)
        self._set_buttons_enabled(False)

        self._auto_thread = QThread()
        self._worker = AutoWorker(self._processor)
        self._worker.moveToThread(self._auto_thread)
        self._auto_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_auto_progress)
        self._worker.error.connect(self._on_auto_error)
        self._worker.finished.connect(self._on_auto_done)
        self._worker.finished.connect(self._auto_thread.quit)
        self._auto_thread.start()

    def _on_auto_progress(self, cur: int, total: int, fname: str):
        self._progress.setValue(cur)
        self._queue.mark_current(cur - 1)
        self._set_status(f"Обробка {cur}/{total}: {fname}")

    def _on_auto_error(self, idx: int, fname: str, msg: str):
        self._queue.mark_error(idx)
        self._set_status(f"Помилка [{idx+1}]: {fname} — {msg}")

    def _on_auto_done(self, count: int):
        self._progress.setVisible(False)
        self._set_buttons_enabled(True)
        self._set_status(f"Готово. Надруковано {count} з {self._processor.total}")
        # Позначаємо всі що не мають статусу
        for i in range(self._queue.count()):
            item = self._queue.item(i)
            if item and not any(item.text().startswith(p) for p in ("✓", "✗")):
                self._queue.mark_done(i)

    # ------------------------------------------------------------------
    # Ручний режим — крокування по черзі
    # ------------------------------------------------------------------

    def _load_next_manual(self):
        # Зберігаємо налаштування поточного файлу перед переходом
        self._store_current_settings()
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
            self._processed = None
            path = self._processor.current_file()
            self._current_path = path
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
    # Допоміжне
    # ------------------------------------------------------------------

    def _update_buttons(self):
        has_queue = bool(self._queue.get_all_paths())
        has_img   = self._orig is not None
        self._btn_print_all.setEnabled(has_queue)
        self._btn_print.setEnabled(has_img)
        self._btn_skip.setEnabled(self._processor.has_next())
        self._btn_autofix.setEnabled(has_img)
        self._btn_save_img.setEnabled(has_img)

    def _set_buttons_enabled(self, enabled: bool):
        for b in (self._btn_autofix, self._btn_print,
                  self._btn_skip, self._btn_print_all):
            b.setEnabled(enabled)

    def _set_status(self, text: str):
        self._status.setText(text)
