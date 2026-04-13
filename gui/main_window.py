"""
Головне вікно програми PhotoPrint.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QButtonGroup, QRadioButton,
    QFileDialog, QMessageBox, QProgressBar, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui  import QDragEnterEvent, QDropEvent

from gui.preview         import PreviewPanel
from gui.queue_view      import QueueView
from gui.controls        import ControlsPanel
from gui.settings_window import SettingsWindow
from batch.batch_processor import BatchProcessor
from processing import pipeline
from utils import file_utils, image_utils
from config import app_settings


# ---------------------------------------------------------------------------
# Worker для авто-режиму
# ---------------------------------------------------------------------------

class AutoWorker(QObject):
    progress = pyqtSignal(int, int, str)
    error    = pyqtSignal(str, str)
    finished = pyqtSignal(int)

    def __init__(self, processor):
        super().__init__()
        self._p = processor

    def run(self):
        count = self._p.run_auto(
            on_progress=lambda c, t, f: self.progress.emit(c, t, f),
            on_error=lambda f, m: self.error.emit(f, m),
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
        self.setAcceptDrops(True)

        self._settings  = app_settings.load()
        self._processor = BatchProcessor(self._settings)
        self._orig:      np.ndarray | None = None   # оригінал поточного файлу
        self._processed: np.ndarray | None = None   # результат після обробки
        self._auto_thread = None
        self._settings_win = SettingsWindow()
        self._settings_win.settings_saved.connect(self._on_settings_saved)

        self._build_ui()
        self._apply_default_mode()
        self._update_buttons()

    # ------------------------------------------------------------------
    # UI
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
        self._queue.files_dropped.connect(self._on_files_dropped)
        self._queue.selection_changed.connect(self._on_queue_selection)

        btn_add   = QPushButton("Додати файли…")
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
        right_scroll.setFixedWidth(230)
        right_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")

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
            "background:#5A8A5A; color:white; border:none; "
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

    # ------------------------------------------------------------------
    # Стиль кнопок
    # ------------------------------------------------------------------

    def _btn_style(self, color="#2E5FA3"):
        return (
            f"QPushButton {{ background:{color}; color:white; border:none; "
            f"border-radius:4px; padding:5px 10px; font-size:13px; }}"
            f"QPushButton:hover {{ background:{color}DD; }}"
            f"QPushButton:pressed {{ background:{color}99; }}"
            f"QPushButton:disabled {{ background:#AAAAAA; color:#EEEEEE; }}"
        )

    # ------------------------------------------------------------------
    # Налаштування
    # ------------------------------------------------------------------

    def _apply_default_mode(self):
        if self._settings.get("default_mode", "auto") == "auto":
            self._radio_auto.setChecked(True)
        else:
            self._radio_manual.setChecked(True)
        self._controls.set_sharpen(self._settings.get("sharpen_strength", 0.4))
        self._controls.set_hdr(self._settings.get("hdr_strength", 0.0))

    def _on_settings_saved(self, s):
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

    def _on_files_dropped(self, paths):
        supported = file_utils.filter_supported(paths)
        if not supported:
            self._set_status("Жоден з файлів не підтримується")
            return
        self._processor.add_files(supported)
        self._set_status(f"Додано {len(supported)} файл(ів). Всього: {self._processor.total}")
        self._update_buttons()

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Оберіть зображення", "",
            "Зображення (*.jpg *.jpeg *.png *.webp *.tiff *.tif *.heic *.heif)"
        )
        if paths:
            self._queue.add_files(paths)
            self._on_files_dropped(paths)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Оберіть папку")
        if folder:
            imgs = file_utils.collect_images_from_folder(folder)
            if imgs:
                self._queue.add_files(imgs)
                self._on_files_dropped(imgs)
            else:
                self._set_status("У папці немає підтримуваних зображень")

    def _clear_queue(self):
        self._queue.clear_queue()
        self._processor.clear()
        self._preview.clear()
        self._orig = None
        self._processed = None
        self._update_buttons()
        self._set_status("Черга очищена")

    def _on_queue_selection(self, path):
        try:
            from core import loader
            img = loader.load(path)
            self._orig = img
            self._processed = None
            prev = image_utils.make_preview(img)
            self._preview.set_before(prev)
            self._preview.set_after(prev)
            self._update_buttons()
        except Exception as e:
            self._set_status(f"Помилка завантаження: {e}")

    # ------------------------------------------------------------------
    # Обробка
    # ------------------------------------------------------------------

    def _do_autofix(self):
        if self._orig is None:
            self._set_status("Спочатку оберіть файл")
            return
        try:
            s = self._settings
            result = pipeline.run_autofix(
                self._orig,
                sharpen_strength=self._controls.values()["sharpen_strength"],
                hdr_strength=self._controls.values()["hdr_strength"],
                use_hdr=s.get("hdr_in_autofix", True),
                use_perspective=s.get("auto_perspective", True),
            )
            if self._controls.values()["grayscale"]:
                result = pipeline.run_grayscale(result)
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            self._update_buttons()
        except Exception as e:
            self._set_status(f"Помилка Auto Fix: {e}")

    def _on_controls_changed(self, vals: dict):
        """Миттєво оновлює прев'ю при зміні будь-якого слайдера."""
        if self._orig is None:
            return
        try:
            base = self._orig
            result = pipeline.run_manual_adjustments(
                base,
                brightness=vals["brightness"],
                contrast=vals["contrast"],
                sharpen_strength=vals["sharpen_strength"],
                hdr_strength=vals["hdr_strength"],
                grayscale=vals["grayscale"],
            )
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
        except Exception as e:
            self._set_status(f"Помилка обробки: {e}")

    def _do_auto_brightness(self):
        if self._orig is None:
            return
        result = pipeline.run_auto_brightness(self._orig)
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))

    def _do_auto_contrast(self):
        if self._orig is None:
            return
        result = pipeline.run_auto_contrast(self._orig)
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))

    def _do_persp_auto(self):
        if self._orig is None:
            return
        result, found = pipeline.run_perspective_auto(self._orig)
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
        self._set_status("Перспективу виправлено" if found else "Документ не знайдено — спробуйте ручну")

    def _do_persp_manual(self):
        if self._orig is None:
            return
        corners = pipeline.detect_corners(self._orig)
        h, w = self._orig.shape[:2]
        from PyQt6.QtCore import QPoint
        if corners is not None:
            pts = [QPoint(int(p[0]), int(p[1])) for p in corners]
        else:
            m = max(20, min(w, h) // 10)
            pts = [QPoint(m, m), QPoint(w - m, m), QPoint(w - m, h - m), QPoint(m, h - m)]
        self._preview.enable_perspective_edit(pts)
        self._set_status("Тягніть кольорові точки для корекції перспективи")

    def _on_persp_pts(self, points):
        if self._orig is None or len(points) != 4:
            return
        try:
            from PyQt6.QtCore import QPoint
            corners = np.array([[p.x(), p.y()] for p in points], dtype=np.float32)
            result = pipeline.run_perspective_manual(self._orig, corners)
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Друк та збереження
    # ------------------------------------------------------------------

    def _do_print_current(self):
        # ВИПРАВЛЕНО: правильна перевірка numpy array
        image = self._processed if self._processed is not None else self._orig
        if image is None:
            self._set_status("Немає зображення для друку")
            return
        try:
            self._processor.set_files(self._queue.get_all_paths())
            self._processor.print_current(image)
            self._set_status("Надруковано")
            self._load_next_manual()
        except Exception as e:
            self._set_status(f"Помилка друку: {e}")

    def _do_skip(self):
        self._processor.skip_current()
        idx = self._processor._index - 1
        if idx >= 0:
            self._queue.mark_skipped(idx)
        self._load_next_manual()

    def _do_save_image(self):
        """Зберігає поточне оброблене зображення — для відладки."""
        image = self._processed if self._processed is not None else self._orig
        if image is None:
            self._set_status("Немає зображення для збереження")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти зображення", "result.jpg",
            "JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            from core import saver
            saver.save(image, path, quality=self._settings.get("jpg_quality", 95))
            self._set_status(f"Збережено: {os.path.basename(path)}")
        except Exception as e:
            self._set_status(f"Помилка збереження: {e}")

    def _do_print_all(self):
        if self._processor.total == 0 and not self._queue.get_all_paths():
            self._set_status("Черга порожня")
            return
        self._processor.set_files(self._queue.get_all_paths())
        if self._radio_auto.isChecked():
            self._start_auto()
        else:
            self._load_next_manual()

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

    def _on_auto_progress(self, cur, total, fname):
        self._progress.setValue(cur)
        self._queue.mark_current(cur - 1)
        self._set_status(f"Обробка {cur}/{total}: {fname}")

    def _on_auto_error(self, fname, msg):
        self._queue.mark_error(self._processor._index)
        self._set_status(f"Помилка: {fname} — {msg}")

    def _on_auto_done(self, count):
        self._progress.setVisible(False)
        self._set_buttons_enabled(True)
        self._set_status(f"Готово. Надруковано {count} з {self._processor.total}")

    def _load_next_manual(self):
        if not self._processor.has_next():
            self._set_status("Всі файли оброблено")
            return
        try:
            idx = self._processor._index
            self._queue.mark_current(idx)
            img = self._processor.load_current()
            self._orig = img
            self._processed = None
            prev = image_utils.make_preview(img)
            self._preview.set_before(prev)
            self._preview.set_after(prev)
            path = self._processor.current_file()
            total = self._processor.total
            self._set_status(f"[{idx + 1}/{total}]  {os.path.basename(path or '')}")
            self._update_buttons()
        except Exception as e:
            self._set_status(f"Помилка завантаження: {e}")

    # ------------------------------------------------------------------
    # Drag & Drop на головне вікно (резервний, основний — у QueueView)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile()]
        supported = file_utils.filter_supported(paths)
        if supported:
            self._queue.add_files(supported)
            self._on_files_dropped(supported)
        event.accept()

    # ------------------------------------------------------------------
    # Допоміжне
    # ------------------------------------------------------------------

    def _update_buttons(self):
        has = self._processor.total > 0 or bool(self._queue.get_all_paths())
        has_img = self._orig is not None
        self._btn_print_all.setEnabled(has)
        self._btn_print.setEnabled(has_img)
        self._btn_skip.setEnabled(self._processor.has_next())
        self._btn_autofix.setEnabled(has_img)
        self._btn_save_img.setEnabled(has_img)

    def _set_buttons_enabled(self, enabled):
        for b in (self._btn_autofix, self._btn_print,
                  self._btn_skip, self._btn_print_all):
            b.setEnabled(enabled)

    def _set_status(self, text):
        self._status.setText(text)
