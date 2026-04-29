"""
Вікно налаштувань — відображає і редагує settings.ini.
Статичне (не модальне): можна тримати відкритим поруч із головним вікном.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox,
    QComboBox
)
from PyQt6.QtCore import pyqtSignal
from config import app_settings

# Константи для layout
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 700
LAYOUT_SPACING = 12
GROUPBOX_STYLE = (
    "QGroupBox { font-weight:bold; border:1px solid #BBBBBB; border-radius:4px; "
    "margin-top:8px; padding-top:14px; background:#FAFAFA; }"
    "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
)

# Константи для Shadow Highlight
SHADOW_MIN = 0.0
SHADOW_MAX = 2.0
SHADOW_STEP = 0.1
SHADOW_DECIMALS = 2

# Константи для Sharpen
SHARPEN_MIN = 0.0
SHARPEN_MAX = 1.0
SHARPEN_STEP = 0.05
SHARPEN_DECIMALS = 2

# Константи для HDR
HDR_MIN = 0.0
HDR_MAX = 1.0
HDR_STEP = 0.05
HDR_DECIMALS = 2

# Константи для класифікації
BW_STD_MIN = 1.0
BW_STD_MAX = 100.0
BW_STD_STEP = 1.0
BW_STD_DECIMALS = 1

EDGE_RATIO_MIN = 0.001
EDGE_RATIO_MAX = 0.5
EDGE_RATIO_STEP = 0.01
EDGE_RATIO_DECIMALS = 3

LINE_COUNT_MIN = 0
LINE_COUNT_MAX = 50

# Константи для авто-різкості
AUTOSHARP_THRESH_MIN = 1.0
AUTOSHARP_THRESH_MAX = 500.0
AUTOSHARP_THRESH_STEP = 5.0
AUTOSHARP_THRESH_DECIMALS = 1

AUTOSHARP_MAX_MIN = 0.1
AUTOSHARP_MAX_MAX = 1.0
AUTOSHARP_MAX_STEP = 0.05
AUTOSHARP_MAX_DECIMALS = 2

# Константи для процентилів
PCT_LOW_MIN = 0.0
PCT_LOW_MAX = 25.0
PCT_LOW_STEP = 1.0
PCT_LOW_DECIMALS = 1

PCT_HIGH_MIN = 75.0
PCT_HIGH_MAX = 100.0
PCT_HIGH_STEP = 1.0
PCT_HIGH_DECIMALS = 1

# Константи для якості JPG
QUALITY_MIN = 50
QUALITY_MAX = 100

# Константи для кнопок
BROWSE_BUTTON_WIDTH = 32


class SettingsWindow(QWidget):
    """Вікно налаштувань. Зміни набирають силу після натискання Зберегти."""

    settings_saved = pyqtSignal(dict)   # нові налаштування

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Налаштування")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1200, 750)
        self.setWindowFlag(self._qt_tool_flag(), True)
        self._build_ui()
        self.load_from_file()

    @staticmethod
    def _qt_tool_flag():
        from PyQt6.QtCore import Qt
        return Qt.WindowType.Tool

    # ------------------------------------------------------------------
    # Побудова UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(LAYOUT_SPACING)

        # === Дві колонки ===
        columns = QHBoxLayout()
        columns.setSpacing(LAYOUT_SPACING)

        # --- Ліва колонка ---
        left = QVBoxLayout()
        left.setSpacing(LAYOUT_SPACING)

        # === Обробка ===
        proc_box = QGroupBox("Обробка")
        proc_box.setStyleSheet(GROUPBOX_STYLE)
        proc_form = QFormLayout(proc_box)

        self._cb_autofix        = QCheckBox()
        self._cb_auto_apply     = QCheckBox()
        self._cb_hdr            = QCheckBox()
        self._cb_perspective    = QCheckBox()
        proc_form.addRow("Auto Fix за замовчуванням:",    self._cb_autofix)
        proc_form.addRow("Авто-застосувати при завантаженні:", self._cb_auto_apply)
        proc_form.addRow("HDR в Auto Fix:",                self._cb_hdr)
        proc_form.addRow("Авто-перспектива:",              self._cb_perspective)

        self._spin_shadow = QDoubleSpinBox()
        self._spin_shadow.setRange(SHADOW_MIN, SHADOW_MAX)
        self._spin_shadow.setSingleStep(SHADOW_STEP)
        self._spin_shadow.setDecimals(SHADOW_DECIMALS)

        self._spin_sharpen = QDoubleSpinBox()
        self._spin_sharpen.setRange(SHARPEN_MIN, SHARPEN_MAX)
        self._spin_sharpen.setSingleStep(SHARPEN_STEP)
        self._spin_sharpen.setDecimals(SHARPEN_DECIMALS)

        self._spin_hdr = QDoubleSpinBox()
        self._spin_hdr.setRange(HDR_MIN, HDR_MAX)
        self._spin_hdr.setSingleStep(HDR_STEP)
        self._spin_hdr.setDecimals(HDR_DECIMALS)

        proc_form.addRow("Висвітлення тіней (0–2):", self._spin_shadow)
        proc_form.addRow("Сила різкості (0–1):", self._spin_sharpen)
        proc_form.addRow("Сила HDR (0–1):",      self._spin_hdr)

        # === Класифікація документів ===
        cls_box = QGroupBox("Класифікація документів")
        cls_box.setStyleSheet(GROUPBOX_STYLE)
        cls_form = QFormLayout(cls_box)

        self._spin_bw_std = QDoubleSpinBox()
        self._spin_bw_std.setRange(BW_STD_MIN, BW_STD_MAX)
        self._spin_bw_std.setSingleStep(BW_STD_STEP)
        self._spin_bw_std.setDecimals(BW_STD_DECIMALS)

        self._spin_edge_ratio = QDoubleSpinBox()
        self._spin_edge_ratio.setRange(EDGE_RATIO_MIN, EDGE_RATIO_MAX)
        self._spin_edge_ratio.setSingleStep(EDGE_RATIO_STEP)
        self._spin_edge_ratio.setDecimals(EDGE_RATIO_DECIMALS)

        self._spin_line_count = QSpinBox()
        self._spin_line_count.setRange(LINE_COUNT_MIN, LINE_COUNT_MAX)

        cls_form.addRow("Поріг std(a,b) для ЧБ:",    self._spin_bw_std)
        cls_form.addRow("Мін. частка країв (0–1):",   self._spin_edge_ratio)
        cls_form.addRow("Мін. кількість ліній:",      self._spin_line_count)

        # === Авто-різкість ===
        sh_box = QGroupBox("Авто-різкість")
        sh_box.setStyleSheet(GROUPBOX_STYLE)
        sh_form = QFormLayout(sh_box)

        self._spin_asharp_thresh = QDoubleSpinBox()
        self._spin_asharp_thresh.setRange(AUTOSHARP_THRESH_MIN, AUTOSHARP_THRESH_MAX)
        self._spin_asharp_thresh.setSingleStep(AUTOSHARP_THRESH_STEP)
        self._spin_asharp_thresh.setDecimals(AUTOSHARP_THRESH_DECIMALS)

        self._spin_asharp_max = QDoubleSpinBox()
        self._spin_asharp_max.setRange(AUTOSHARP_MAX_MIN, AUTOSHARP_MAX_MAX)
        self._spin_asharp_max.setSingleStep(AUTOSHARP_MAX_STEP)
        self._spin_asharp_max.setDecimals(AUTOSHARP_MAX_DECIMALS)

        sh_form.addRow("Поріг Laplacian variance:",  self._spin_asharp_thresh)
        sh_form.addRow("Макс. сила різкості (0–1):", self._spin_asharp_max)

        # === Авто-яскравість/контраст ===
        pct_box = QGroupBox("Авто-яскравість / контраст")
        pct_box.setStyleSheet(GROUPBOX_STYLE)
        pct_form = QFormLayout(pct_box)

        self._spin_pct_low = QDoubleSpinBox()
        self._spin_pct_low.setRange(PCT_LOW_MIN, PCT_LOW_MAX)
        self._spin_pct_low.setSingleStep(PCT_LOW_STEP)
        self._spin_pct_low.setDecimals(PCT_LOW_DECIMALS)

        self._spin_pct_high = QDoubleSpinBox()
        self._spin_pct_high.setRange(PCT_HIGH_MIN, PCT_HIGH_MAX)
        self._spin_pct_high.setSingleStep(PCT_HIGH_STEP)
        self._spin_pct_high.setDecimals(PCT_HIGH_DECIMALS)

        pct_form.addRow("Нижній процентиль (%):",  self._spin_pct_low)
        pct_form.addRow("Верхній процентиль (%):", self._spin_pct_high)

        left.addWidget(proc_box)
        left.addWidget(cls_box)
        left.addWidget(sh_box)
        left.addWidget(pct_box)
        left.addStretch()

        # --- Права колонка ---
        right = QVBoxLayout()
        right.setSpacing(LAYOUT_SPACING)

        # === Формат виходу ===
        out_box = QGroupBox("Формат виходу")
        out_box.setStyleSheet(GROUPBOX_STYLE)
        out_form = QFormLayout(out_box)

        self._combo_color_mode = QComboBox()
        self._combo_color_mode.addItem("Авто (за типом документа)", "auto")
        self._combo_color_mode.addItem("Кольоровий", "color")
        self._combo_color_mode.addItem("Чорно-білий (напівтони)", "grayscale")
        self._combo_color_mode.addItem("Чорно-білий (бінаризація)", "binary")

        out_form.addRow("Формат виходу:", self._combo_color_mode)

        self._cb_bw_binary = QCheckBox("Адаптивна бінаризація")
        out_form.addRow("Ч-б бінаризація:", self._cb_bw_binary)

        # === Збереження ===
        save_box = QGroupBox("Збереження")
        save_box.setStyleSheet(GROUPBOX_STYLE)
        save_form = QFormLayout(save_box)

        self._spin_quality = QSpinBox()
        self._spin_quality.setRange(QUALITY_MIN, QUALITY_MAX)

        self._edit_folder = QLineEdit()
        self._edit_folder.setPlaceholderText("(порожньо = не зберігати)")
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(BROWSE_BUTTON_WIDTH)
        btn_browse.clicked.connect(self._browse_folder)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self._edit_folder)
        folder_row.addWidget(btn_browse)

        save_form.addRow("Якість JPG (50–100):", self._spin_quality)
        save_form.addRow("Папка збереження:",    folder_row)

        # === Принтер ===
        print_box = QGroupBox("Принтер")
        print_box.setStyleSheet(GROUPBOX_STYLE)
        print_form = QFormLayout(print_box)

        self._edit_printer = QLineEdit()
        self._edit_printer.setPlaceholderText("priPrinter")
        print_form.addRow("Назва принтера:", self._edit_printer)

        # === Режим запуску ===
        mode_box = QGroupBox("Режим запуску")
        mode_box.setStyleSheet(GROUPBOX_STYLE)
        mode_form = QFormLayout(mode_box)
        self._cb_default_auto = QCheckBox("Авто (інакше — Ручний)")
        mode_form.addRow(self._cb_default_auto)

        right.addWidget(out_box)
        right.addWidget(save_box)
        right.addWidget(print_box)
        right.addWidget(mode_box)
        right.addStretch()

        # === Збираємо колонки ===
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        root.addLayout(columns)

        # === Кнопки ===
        btn_row = QHBoxLayout()
        btn_save   = QPushButton("Зберегти")
        btn_cancel = QPushButton("Скасувати")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._save)
        btn_cancel.clicked.connect(self.hide)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_cancel)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Завантаження / збереження
    # ------------------------------------------------------------------

    def load_from_file(self):
        s = app_settings.load()
        self._apply_settings(s)

    def _apply_settings(self, s: dict):
        self._cb_autofix.setChecked(s.get("autofix_enabled", True))
        self._cb_auto_apply.setChecked(s.get("auto_apply_autofix", True))
        self._cb_hdr.setChecked(s.get("hdr_in_autofix", True))
        self._cb_perspective.setChecked(s.get("auto_perspective", False))
        self._spin_shadow.setValue(s.get("shadow_highlight_strength", 0.0))
        self._spin_sharpen.setValue(s.get("sharpen_strength", 0.4))
        self._spin_hdr.setValue(s.get("hdr_strength", 0.5))

        self._spin_bw_std.setValue(s.get("classify_bw_std_thresh", 20.0))
        self._spin_edge_ratio.setValue(s.get("classify_edge_ratio_min", 0.03))
        self._spin_line_count.setValue(s.get("classify_line_count_min", 3))

        self._spin_asharp_thresh.setValue(s.get("autosharp_threshold", 80.0))
        self._spin_asharp_max.setValue(s.get("autosharp_max_strength", 0.7))

        self._spin_pct_low.setValue(s.get("auto_percentile_low", 5.0))
        self._spin_pct_high.setValue(s.get("auto_percentile_high", 95.0))

        self._cb_bw_binary.setChecked(s.get("bw_binary", False))

        # Формат виходу
        color_mode = s.get("output_color_mode", "auto")
        idx = self._combo_color_mode.findData(color_mode)
        if idx >= 0:
            self._combo_color_mode.setCurrentIndex(idx)

        self._spin_quality.setValue(s.get("jpg_quality", 95))
        self._edit_folder.setText(s.get("save_folder", ""))
        self._edit_printer.setText(s.get("printer_name", "priPrinter"))
        self._cb_default_auto.setChecked(s.get("default_mode", "auto") == "auto")

    def _collect_settings(self) -> dict:
        return {
            "autofix_enabled":    self._cb_autofix.isChecked(),
            "auto_apply_autofix": self._cb_auto_apply.isChecked(),
            "hdr_in_autofix":     self._cb_hdr.isChecked(),
            "auto_perspective":   self._cb_perspective.isChecked(),
            "shadow_highlight_strength": self._spin_shadow.value(),
            "sharpen_strength":   self._spin_sharpen.value(),
            "hdr_strength":       self._spin_hdr.value(),

            "classify_bw_std_thresh":   self._spin_bw_std.value(),
            "classify_edge_ratio_min":  self._spin_edge_ratio.value(),
            "classify_line_count_min":  self._spin_line_count.value(),

            "autosharp_threshold":    self._spin_asharp_thresh.value(),
            "autosharp_max_strength": self._spin_asharp_max.value(),

            "auto_percentile_low":  self._spin_pct_low.value(),
            "auto_percentile_high": self._spin_pct_high.value(),

            "bw_binary": self._cb_bw_binary.isChecked(),

            "output_color_mode": self._combo_color_mode.currentData(),
            "jpg_quality":       self._spin_quality.value(),
            "save_folder":       self._edit_folder.text().strip(),
            "printer_name":      self._edit_printer.text().strip(),
            "default_mode":      "auto" if self._cb_default_auto.isChecked() else "manual",
        }

    def _save(self):
        s = self._collect_settings()
        try:
            app_settings.save(s)
            self.settings_saved.emit(s)
            self.hide()
        except Exception as exc:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{exc}")

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Оберіть папку збереження")
        if folder:
            self._edit_folder.setText(folder)
