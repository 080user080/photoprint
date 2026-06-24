"""
Вікно налаштувань — відображає і редагує settings.ini.
Статичне (не модальне): можна тримати відкритим поруч із головним вікном.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox,
    QComboBox, QListWidget, QListWidgetItem, QStackedWidget,
    QScrollArea
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
from config import app_settings

# Константи для layout
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 700
LAYOUT_SPACING = 16
GROUPBOX_STYLE = (
    "QGroupBox { font-weight:bold; border:1px solid #BBBBBB; border-radius:4px; "
    "margin-top:8px; padding-top:14px; background:#FAFAFA; }"
    "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
)
SIDEBAR_WIDTH = 200
SPINBOX_MIN_WIDTH = 100

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

# Константи для Shadow Remove
SHADOW_DETECT_THRESHOLD_MIN = 20.0
SHADOW_DETECT_THRESHOLD_MAX = 200.0
SHADOW_DETECT_THRESHOLD_STEP = 5.0

SHADOW_DETECT_RATIO_MIN = 0.05
SHADOW_DETECT_RATIO_MAX = 0.80
SHADOW_DETECT_RATIO_STEP = 0.05
SHADOW_DETECT_RATIO_DECIMALS = 2

SHADOW_COARSE_BLEND_MIN = 0.0
SHADOW_COARSE_BLEND_MAX = 1.0
SHADOW_COARSE_BLEND_STEP = 0.1
SHADOW_COARSE_BLEND_DECIMALS = 1

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

# Константи для Auto Fix контрасту
AUTOFIX_CONTRAST_MIN = 0.0
AUTOFIX_CONTRAST_MAX = 1.0
AUTOFIX_CONTRAST_STEP = 0.05
AUTOFIX_CONTRAST_DECIMALS = 2

# Константи для якості JPG
QUALITY_MIN = 50
QUALITY_MAX = 100

# Константи для кнопок
BROWSE_BUTTON_WIDTH = 32

# === Пресети стратегій ===
PRESETS = {
    "doc_bw": {
        "label": "Документ (чб)",
        "steps": ["shadow_remove", "perspective", "brightness", "contrast", "sharpen", "grayscale", "white_background"],
    },
    "doc_color": {
        "label": "Документ (кольоровий)",
        "steps": ["shadow_remove", "perspective", "brightness", "contrast", "sharpen", "white_background"],
    },
    "photo": {
        "label": "Фото",
        "steps": ["perspective", "hdr", "sharpen"],
    },
    "geometry": {
        "label": "Тільки геометрія",
        "steps": ["perspective"],
    },
    "custom": {
        "label": "Власний",
        "steps": None,
    },
}

# Фіксований порядок кроків обробки
PIPELINE_STEPS_FIXED_ORDER = [
    ("shadow_remove",    "Видалення тіней"),
    ("perspective",      "Авто-перспектива"),
    ("brightness",       "Авто-яскравість"),
    ("contrast",         "Авто-контраст"),
    ("hdr",              "HDR"),
    ("sharpen",          "Різкість"),
    ("grayscale",        "Grayscale / бінаризація"),
    ("white_background", "Білий фон"),
]

# === Список розділів ===
SECTIONS = [
    ("autofix",      "Auto Fix"),
    ("shadowremove", "Видалення тіней"),
    ("classify",     "Класифікація"),
    ("sharpen",      "Авто-різкість"),
    ("brightcontr",  "Яскравість/контраст"),
    ("strategy",     "Стратегія обробки"),
    ("format",       "Формат/Збереження"),
    ("printer",      "Принтер/Режим"),
]


class StrategyPresetWidget(QWidget):
    """Віджет вибору пресета стратегії обробки."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._combo = QComboBox()
        for key, preset in PRESETS.items():
            self._combo.addItem(preset["label"], key)
        layout.addWidget(self._combo)

        self._list = QListWidget()
        self._list.setMaximumHeight(200)
        self._checkboxes = []
        for key, label in PIPELINE_STEPS_FIXED_ORDER:
            item = QListWidgetItem()
            cb = QCheckBox(label)
            cb.setChecked(True)
            item.setSizeHint(cb.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, cb)
            self._checkboxes.append((key, cb))
        layout.addWidget(self._list)

        self._combo.currentIndexChanged.connect(self._on_preset_changed)
        self._on_preset_changed(0)

    def _on_preset_changed(self, index):
        key = self._combo.currentData()
        self._list.setVisible(key == "custom")

    def get_preset(self) -> str:
        return self._combo.currentData()

    def get_enabled_steps(self) -> list[str]:
        key = self._combo.currentData()
        if key != "custom":
            steps = PRESETS.get(key, {}).get("steps", [])
            return steps if steps else []
        enabled = []
        for step_key, cb in self._checkboxes:
            if cb.isChecked():
                enabled.append(step_key)
        return enabled

    def set_state(self, preset: str, enabled: list[str] | None = None):
        idx = self._combo.findData(preset)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        if preset == "custom" and enabled is not None:
            for step_key, cb in self._checkboxes:
                cb.setChecked(step_key in enabled)
        self._on_preset_changed(self._combo.currentIndex())


def _make_scroll_page(widget: QWidget) -> QWidget:
    """Загорнути віджет у QScrollArea всередині QWidget."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    layout.addWidget(scroll)
    return page


def _set_spinbox_minw(spin):
    """Встановити мінімальну ширину для спін-бокса."""
    spin.setMinimumWidth(SPINBOX_MIN_WIDTH)


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
    # Побудова UI — сайдбар + сторінки
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(LAYOUT_SPACING)

        # Основний горизонтальний layout: сайдбар + контент
        hbox = QHBoxLayout()
        hbox.setSpacing(LAYOUT_SPACING)

        # --- Лівий сайдбар ---
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(SIDEBAR_WIDTH)
        self._sidebar.setSpacing(2)
        for key, label in SECTIONS:
            self._sidebar.addItem(label)
        self._sidebar.setCurrentRow(0)

        # --- Правий стек ---
        self._stack = QStackedWidget()

        # Сторінки
        self._stack.addWidget(_make_scroll_page(self._page_autofix()))
        self._stack.addWidget(_make_scroll_page(self._page_shadow_remove()))
        self._stack.addWidget(_make_scroll_page(self._page_classify()))
        self._stack.addWidget(_make_scroll_page(self._page_autosharp()))
        self._stack.addWidget(_make_scroll_page(self._page_brightness_contrast()))
        self._stack.addWidget(_make_scroll_page(self._page_strategy()))
        self._stack.addWidget(_make_scroll_page(self._page_format_save()))
        self._stack.addWidget(_make_scroll_page(self._page_printer_mode()))

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)

        hbox.addWidget(self._sidebar)
        hbox.addWidget(self._stack, 1)
        root.addLayout(hbox)

        # --- Кнопки внизу ---
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
    # Сторінки
    # ------------------------------------------------------------------

    def _page_autofix(self) -> QWidget:
        """Сторінка Auto Fix — основні параметри обробки."""
        box = QGroupBox("Auto Fix")
        box.setStyleSheet(GROUPBOX_STYLE)
        form = QFormLayout(box)

        self._cb_autofix        = QCheckBox()
        self._cb_auto_apply     = QCheckBox()
        self._cb_hdr            = QCheckBox()
        self._cb_perspective    = QCheckBox()
        form.addRow("Auto Fix за замовчуванням:",    self._cb_autofix)
        form.addRow("Авто-застосувати при завантаженні:", self._cb_auto_apply)
        form.addRow("HDR в Auto Fix:",                self._cb_hdr)
        form.addRow("Авто-перспектива:",              self._cb_perspective)

        self._spin_shadow = QDoubleSpinBox()
        self._spin_shadow.setRange(SHADOW_MIN, SHADOW_MAX)
        self._spin_shadow.setSingleStep(SHADOW_STEP)
        self._spin_shadow.setDecimals(SHADOW_DECIMALS)
        _set_spinbox_minw(self._spin_shadow)

        self._spin_sharpen = QDoubleSpinBox()
        self._spin_sharpen.setRange(SHARPEN_MIN, SHARPEN_MAX)
        self._spin_sharpen.setSingleStep(SHARPEN_STEP)
        self._spin_sharpen.setDecimals(SHARPEN_DECIMALS)
        _set_spinbox_minw(self._spin_sharpen)

        self._spin_hdr = QDoubleSpinBox()
        self._spin_hdr.setRange(HDR_MIN, HDR_MAX)
        self._spin_hdr.setSingleStep(HDR_STEP)
        self._spin_hdr.setDecimals(HDR_DECIMALS)
        _set_spinbox_minw(self._spin_hdr)

        self._spin_autofix_contrast = QDoubleSpinBox()
        self._spin_autofix_contrast.setRange(AUTOFIX_CONTRAST_MIN, AUTOFIX_CONTRAST_MAX)
        self._spin_autofix_contrast.setSingleStep(AUTOFIX_CONTRAST_STEP)
        self._spin_autofix_contrast.setDecimals(AUTOFIX_CONTRAST_DECIMALS)
        _set_spinbox_minw(self._spin_autofix_contrast)

        self._combo_contrast_mode = QComboBox()
        self._combo_contrast_mode.addItem("Лінійний (класичний)", "linear")
        self._combo_contrast_mode.addItem("Перцентильне розтягнення", "percentile")
        self._combo_contrast_mode.addItem("S-подібна крива", "s_curve")
        self._combo_contrast_mode.addItem("Локальний адаптивний", "adaptive")

        form.addRow("Метод контрасту:", self._combo_contrast_mode)
        form.addRow("Контраст Auto Fix (0–1):", self._spin_autofix_contrast)
        form.addRow("Висвітлення тіней (0–2):", self._spin_shadow)
        form.addRow("Сила різкості (0–1):", self._spin_sharpen)
        form.addRow("Сила HDR (0–1):",      self._spin_hdr)

        return box

    def _page_shadow_remove(self) -> QWidget:
        """Сторінка видалення тіней."""
        box = QGroupBox("Видалення тіней")
        box.setStyleSheet(GROUPBOX_STYLE)
        form = QFormLayout(box)

        self._cb_shadow_remove = QCheckBox()
        form.addRow("Видалення тіней увімкнено:", self._cb_shadow_remove)

        self._spin_shadow_detect_threshold = QDoubleSpinBox()
        self._spin_shadow_detect_threshold.setRange(SHADOW_DETECT_THRESHOLD_MIN, SHADOW_DETECT_THRESHOLD_MAX)
        self._spin_shadow_detect_threshold.setSingleStep(SHADOW_DETECT_THRESHOLD_STEP)
        self._spin_shadow_detect_threshold.setDecimals(0)
        _set_spinbox_minw(self._spin_shadow_detect_threshold)
        form.addRow("Поріг темних ділянок p5 (0-255):", self._spin_shadow_detect_threshold)

        self._spin_shadow_detect_ratio = QDoubleSpinBox()
        self._spin_shadow_detect_ratio.setRange(SHADOW_DETECT_RATIO_MIN, SHADOW_DETECT_RATIO_MAX)
        self._spin_shadow_detect_ratio.setSingleStep(SHADOW_DETECT_RATIO_STEP)
        self._spin_shadow_detect_ratio.setDecimals(SHADOW_DETECT_RATIO_DECIMALS)
        _set_spinbox_minw(self._spin_shadow_detect_ratio)
        form.addRow("Поріг відношення p5/p95 (0-1):", self._spin_shadow_detect_ratio)

        self._spin_shadow_coarse_blend = QDoubleSpinBox()
        self._spin_shadow_coarse_blend.setRange(SHADOW_COARSE_BLEND_MIN, SHADOW_COARSE_BLEND_MAX)
        self._spin_shadow_coarse_blend.setSingleStep(SHADOW_COARSE_BLEND_STEP)
        self._spin_shadow_coarse_blend.setDecimals(SHADOW_COARSE_BLEND_DECIMALS)
        _set_spinbox_minw(self._spin_shadow_coarse_blend)
        form.addRow("2-й прохід для кольорових (0=вимк, 1=повний):", self._spin_shadow_coarse_blend)

        return box

    def _page_classify(self) -> QWidget:
        """Сторінка класифікації документів."""
        box = QGroupBox("Класифікація документів")
        box.setStyleSheet(GROUPBOX_STYLE)
        form = QFormLayout(box)

        self._spin_bw_std = QDoubleSpinBox()
        self._spin_bw_std.setRange(BW_STD_MIN, BW_STD_MAX)
        self._spin_bw_std.setSingleStep(BW_STD_STEP)
        self._spin_bw_std.setDecimals(BW_STD_DECIMALS)
        _set_spinbox_minw(self._spin_bw_std)

        self._spin_edge_ratio = QDoubleSpinBox()
        self._spin_edge_ratio.setRange(EDGE_RATIO_MIN, EDGE_RATIO_MAX)
        self._spin_edge_ratio.setSingleStep(EDGE_RATIO_STEP)
        self._spin_edge_ratio.setDecimals(EDGE_RATIO_DECIMALS)
        _set_spinbox_minw(self._spin_edge_ratio)

        self._spin_line_count = QSpinBox()
        self._spin_line_count.setRange(LINE_COUNT_MIN, LINE_COUNT_MAX)
        _set_spinbox_minw(self._spin_line_count)

        form.addRow("Поріг std(a,b) для ЧБ:",    self._spin_bw_std)
        form.addRow("Мін. частка країв (0–1):",   self._spin_edge_ratio)
        form.addRow("Мін. кількість ліній:",      self._spin_line_count)

        return box

    def _page_autosharp(self) -> QWidget:
        """Сторінка авто-різкості."""
        box = QGroupBox("Авто-різкість")
        box.setStyleSheet(GROUPBOX_STYLE)
        form = QFormLayout(box)

        self._spin_asharp_thresh = QDoubleSpinBox()
        self._spin_asharp_thresh.setRange(AUTOSHARP_THRESH_MIN, AUTOSHARP_THRESH_MAX)
        self._spin_asharp_thresh.setSingleStep(AUTOSHARP_THRESH_STEP)
        self._spin_asharp_thresh.setDecimals(AUTOSHARP_THRESH_DECIMALS)
        _set_spinbox_minw(self._spin_asharp_thresh)

        self._spin_asharp_max = QDoubleSpinBox()
        self._spin_asharp_max.setRange(AUTOSHARP_MAX_MIN, AUTOSHARP_MAX_MAX)
        self._spin_asharp_max.setSingleStep(AUTOSHARP_MAX_STEP)
        self._spin_asharp_max.setDecimals(AUTOSHARP_MAX_DECIMALS)
        _set_spinbox_minw(self._spin_asharp_max)

        form.addRow("Поріг Laplacian variance:",  self._spin_asharp_thresh)
        form.addRow("Макс. сила різкості (0–1):", self._spin_asharp_max)

        return box

    def _page_brightness_contrast(self) -> QWidget:
        """Сторінка яскравість/контраст."""
        box = QGroupBox("Авто-яскравість / контраст")
        box.setStyleSheet(GROUPBOX_STYLE)
        form = QFormLayout(box)

        self._spin_pct_low = QDoubleSpinBox()
        self._spin_pct_low.setRange(PCT_LOW_MIN, PCT_LOW_MAX)
        self._spin_pct_low.setSingleStep(PCT_LOW_STEP)
        self._spin_pct_low.setDecimals(PCT_LOW_DECIMALS)
        _set_spinbox_minw(self._spin_pct_low)

        self._spin_pct_high = QDoubleSpinBox()
        self._spin_pct_high.setRange(PCT_HIGH_MIN, PCT_HIGH_MAX)
        self._spin_pct_high.setSingleStep(PCT_HIGH_STEP)
        self._spin_pct_high.setDecimals(PCT_HIGH_DECIMALS)
        _set_spinbox_minw(self._spin_pct_high)

        form.addRow("Відсікання тіней (%):",   self._spin_pct_low)
        form.addRow("Відсікання світла (%):",  self._spin_pct_high)

        return box

    def _page_strategy(self) -> QWidget:
        """Сторінка стратегії обробки."""
        box = QGroupBox("Стратегія обробки")
        box.setStyleSheet(GROUPBOX_STYLE)
        layout = QVBoxLayout(box)
        self._preset_widget = StrategyPresetWidget()
        layout.addWidget(self._preset_widget)
        return box

    def _page_format_save(self) -> QWidget:
        """Сторінка формату виходу та збереження."""
        vbox = QVBoxLayout()
        vbox.setSpacing(LAYOUT_SPACING)

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
        _set_spinbox_minw(self._spin_quality)

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

        # Збираємо в один віджет
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT_SPACING)
        layout.addWidget(out_box)
        layout.addWidget(save_box)
        layout.addStretch()
        return container

    def _page_printer_mode(self) -> QWidget:
        """Сторінка принтера та режиму запуску."""
        vbox = QVBoxLayout()
        vbox.setSpacing(LAYOUT_SPACING)

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
        self._cb_minimize_to_tray = QCheckBox("Згортати в трей при закритті")
        mode_form.addRow(self._cb_minimize_to_tray)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT_SPACING)
        layout.addWidget(print_box)
        layout.addWidget(mode_box)
        layout.addStretch()
        return container

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
        self._spin_autofix_contrast.setValue(s.get("autofix_contrast", 0.15))
        self._spin_shadow.setValue(s.get("shadow_highlight_strength", 0.0))
        self._spin_sharpen.setValue(s.get("sharpen_strength", 0.4))
        self._spin_hdr.setValue(s.get("hdr_strength", 0.5))

        # Параметри видалення тіней
        self._cb_shadow_remove.setChecked(s.get("shadow_remove_enabled", True))
        self._spin_shadow_detect_threshold.setValue(s.get("shadow_detect_threshold", 80.0))
        self._spin_shadow_detect_ratio.setValue(s.get("shadow_detect_ratio", 0.3))
        self._spin_shadow_coarse_blend.setValue(s.get("shadow_coarse_blend_color", 0.0))

        self._spin_bw_std.setValue(s.get("classify_bw_std_thresh", 20.0))
        self._spin_edge_ratio.setValue(s.get("classify_edge_ratio_min", 0.03))
        self._spin_line_count.setValue(s.get("classify_line_count_min", 3))

        self._spin_asharp_thresh.setValue(s.get("autosharp_threshold", 80.0))
        self._spin_asharp_max.setValue(s.get("autosharp_max_strength", 0.7))

        self._spin_pct_low.setValue(s.get("auto_percentile_low", 5.0))
        self._spin_pct_high.setValue(s.get("auto_percentile_high", 95.0))

        self._cb_bw_binary.setChecked(s.get("bw_binary", False))

        # Метод контрасту
        contrast_mode = s.get("contrast_mode", "linear")
        idx = self._combo_contrast_mode.findData(contrast_mode)
        if idx >= 0:
            self._combo_contrast_mode.setCurrentIndex(idx)

        # Формат виходу
        color_mode = s.get("output_color_mode", "auto")
        idx = self._combo_color_mode.findData(color_mode)
        if idx >= 0:
            self._combo_color_mode.setCurrentIndex(idx)

        self._spin_quality.setValue(s.get("jpg_quality", 95))
        self._edit_folder.setText(s.get("save_folder", ""))
        self._edit_printer.setText(s.get("printer_name", "priPrinter"))
        self._cb_default_auto.setChecked(s.get("default_mode", "auto") == "auto")
        self._cb_minimize_to_tray.setChecked(s.get("minimize_to_tray", False))

        # Пресет стратегії
        preset = s.get("pipeline_preset", "doc_bw")
        enabled_str = s.get("pipeline_steps_enabled", "")
        enabled = [k.strip() for k in enabled_str.split(",") if k.strip()] if enabled_str else None
        self._preset_widget.set_state(preset, enabled)

    def _collect_settings(self) -> dict:
        return {
            "autofix_enabled":    self._cb_autofix.isChecked(),
            "auto_apply_autofix": self._cb_auto_apply.isChecked(),
            "hdr_in_autofix":     self._cb_hdr.isChecked(),
            "auto_perspective":   self._cb_perspective.isChecked(),
            "autofix_contrast":   self._spin_autofix_contrast.value(),
            "shadow_highlight_strength": self._spin_shadow.value(),
            "shadow_remove_enabled":     self._cb_shadow_remove.isChecked(),
            "shadow_detect_threshold":   self._spin_shadow_detect_threshold.value(),
            "shadow_detect_ratio":       self._spin_shadow_detect_ratio.value(),
            "shadow_coarse_blend_color": self._spin_shadow_coarse_blend.value(),
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

            "contrast_mode": self._combo_contrast_mode.currentData(),

            "output_color_mode": self._combo_color_mode.currentData(),
            "jpg_quality":       self._spin_quality.value(),
            "save_folder":       self._edit_folder.text().strip(),
            "printer_name":      self._edit_printer.text().strip(),
            "pipeline_preset":        self._preset_widget.get_preset(),
            "pipeline_steps_enabled":  ",".join(self._preset_widget.get_enabled_steps()),
            "default_mode":      "auto" if self._cb_default_auto.isChecked() else "manual",
            "minimize_to_tray":  self._cb_minimize_to_tray.isChecked(),
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