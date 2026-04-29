"""
Панель інструментів: яскравість, контраст, різкість, HDR, Ч/Б, перспектива.
"""

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal

# Константи для слайдерів
SLIDER_SCALE = 100
LABEL_WIDTH = 80
VALUE_LABEL_WIDTH = 45
AUTO_BUTTON_WIDTH = 50
AUTO_BUTTON_HEIGHT = 24

# Константи для діапазонів слайдерів
SHADOW_HIGHLIGHT_MIN = 0.0
SHADOW_HIGHLIGHT_MAX = 2.0
SHADOW_HIGHLIGHT_DEFAULT = 0.0

BRIGHTNESS_MIN = -1.0
BRIGHTNESS_MAX = 1.0
BRIGHTNESS_DEFAULT = 0.0

CONTRAST_MIN = -1.0
CONTRAST_MAX = 1.0
CONTRAST_DEFAULT = 0.0

SHARPEN_MIN = 0.0
SHARPEN_MAX = 1.0
SHARPEN_DEFAULT = 0.4

HDR_MIN = 0.0
HDR_MAX = 1.0
HDR_DEFAULT = 0.0


class _SliderRow(QWidget):
    changed = pyqtSignal(float)
    auto_clicked = pyqtSignal()

    def __init__(self, label: str, min_val: float, max_val: float, default: float, show_auto: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._scale = SLIDER_SCALE

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(LABEL_WIDTH)
        lbl.setStyleSheet("color: #111111; font-size: 13px;")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val * self._scale), int(max_val * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_change)

        self._val_lbl = QLabel(f"{default:.2f}")
        self._val_lbl.setFixedWidth(VALUE_LABEL_WIDTH)
        self._val_lbl.setStyleSheet("color: #111111; font-size: 13px;")

        layout.addWidget(lbl)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._val_lbl)

        if show_auto:
            btn = QPushButton("Авто")
            btn.setFixedWidth(AUTO_BUTTON_WIDTH)
            btn.setFixedHeight(AUTO_BUTTON_HEIGHT)
            btn.setStyleSheet(
                "background:#4A7FC1; color:white; font-size:12px;"
                "border-radius:3px; padding:0px;"
            )
            btn.clicked.connect(self.auto_clicked)
            layout.addWidget(btn)

    def _on_change(self, raw: int) -> None:
        v = raw / self._scale
        self._val_lbl.setText(f"{v:.2f}")
        self.changed.emit(v)

    def value(self) -> float:
        return self._slider.value() / self._scale

    def set_value(self, v: float, silent: bool = False) -> None:
        if silent:
            self._slider.blockSignals(True)
        self._slider.setValue(int(v * self._scale))
        self._val_lbl.setText(f"{v:.2f}")
        if silent:
            self._slider.blockSignals(False)

    def reset(self) -> None:
        self.set_value(0.0, silent=True)


class ControlsPanel(QWidget):
    changed = pyqtSignal(dict)
    auto_brightness_clicked   = pyqtSignal()
    auto_contrast_clicked     = pyqtSignal()
    auto_sharpen_clicked      = pyqtSignal()
    perspective_auto_clicked  = pyqtSignal()
    perspective_manual_clicked = pyqtSignal()
    perspective_reset_clicked  = pyqtSignal()
    reset_all_clicked         = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()

    def _group_style(self) -> str:
        return (
            "QGroupBox { border:1px solid #BBBBBB; border-radius:4px; "
            "margin-top:6px; background:#FFFFFF; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; "
            "padding:0 4px; color:#1F3864; font-weight:bold; font-size:12px; }"
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Перший ряд слайдерів
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        # Висвітлення тіней
        self._shadow_highlight = _SliderRow("Тіні", SHADOW_HIGHLIGHT_MIN, SHADOW_HIGHLIGHT_MAX, SHADOW_HIGHLIGHT_DEFAULT)
        self._shadow_highlight.changed.connect(self._emit)
        row1.addWidget(self._shadow_highlight)

        # Яскравість
        self._brightness = _SliderRow("Яскравість", BRIGHTNESS_MIN, BRIGHTNESS_MAX, BRIGHTNESS_DEFAULT, show_auto=True)
        self._brightness.changed.connect(self._emit)
        self._brightness.auto_clicked.connect(self.auto_brightness_clicked)
        row1.addWidget(self._brightness)

        # Контраст
        self._contrast = _SliderRow("Контраст", CONTRAST_MIN, CONTRAST_MAX, CONTRAST_DEFAULT, show_auto=True)
        self._contrast.changed.connect(self._emit)
        self._contrast.auto_clicked.connect(self.auto_contrast_clicked)
        row1.addWidget(self._contrast)

        # Другий ряд слайдерів
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        # Різкість
        self._sharpen = _SliderRow("Різкість", SHARPEN_MIN, SHARPEN_MAX, SHARPEN_DEFAULT, show_auto=True)
        self._sharpen.changed.connect(self._emit)
        self._sharpen.auto_clicked.connect(self.auto_sharpen_clicked)
        row2.addWidget(self._sharpen)

        # HDR
        self._hdr = _SliderRow("HDR", HDR_MIN, HDR_MAX, HDR_DEFAULT)
        self._hdr.changed.connect(self._emit)
        row2.addWidget(self._hdr)

        # Ряд під слайдерами: Ч/Б та перспектива
        misc_row = QHBoxLayout()
        misc_row.setSpacing(8)

        self._cb_bw = QCheckBox("Чорно-білий")
        self._cb_bw.setStyleSheet("color:#111111; font-size:13px;")
        self._cb_bw.toggled.connect(self._emit)
        misc_row.addWidget(self._cb_bw)

        btn_pa = QPushButton("Авто-перспектива")
        btn_pa.setStyleSheet("background:#6B8E23; color:white; border:none; border-radius:3px; padding:5px 10px; font-size:12px;")
        btn_pa.clicked.connect(self.perspective_auto_clicked)
        misc_row.addWidget(btn_pa)

        btn_pm = QPushButton("Ручна перспектива")
        btn_pm.setStyleSheet("background:#6B8E23; color:white; border:none; border-radius:3px; padding:5px 10px; font-size:12px;")
        btn_pm.clicked.connect(self.perspective_manual_clicked)
        misc_row.addWidget(btn_pm)

        btn_persp_reset = QPushButton("Скинути перспективу")
        btn_persp_reset.setStyleSheet("background:#CD853F; color:white; border:none; border-radius:3px; padding:5px 10px; font-size:12px;")
        btn_persp_reset.clicked.connect(self.perspective_reset_clicked)
        misc_row.addWidget(btn_persp_reset)

        # Скинути слайдери
        btn_reset = QPushButton("Скинути слайдери")
        btn_reset.setStyleSheet(
            "background:#888888; color:white; border:none; border-radius:3px; padding:5px 10px; font-size:12px;"
        )
        btn_reset.clicked.connect(self.reset_all)
        misc_row.addWidget(btn_reset)

        misc_row.addStretch()

        root.addLayout(row1)
        root.addLayout(row2)
        root.addLayout(misc_row)

    def _emit(self, *_) -> None:
        self.changed.emit(self.values())

    def values(self) -> Dict[str, Any]:
        return {
            "shadow_highlight": self._shadow_highlight.value(),
            "brightness":       self._brightness.value(),
            "contrast":         self._contrast.value(),
            "sharpen_strength": self._sharpen.value(),
            "hdr_strength":     self._hdr.value(),
            "grayscale":        self._cb_bw.isChecked(),
        }

    def reset_all(self) -> None:
        for w in (self._shadow_highlight, self._brightness, self._contrast, self._sharpen, self._hdr):
            w.reset()
        self._cb_bw.setChecked(False)
        # Emit сигнал для скидання базового зображення
        self.reset_all_clicked.emit()
        # Emit сигнал для оновлення прев'ю
        self._emit()

    def set_sharpen(self, v: float, silent: bool = True) -> None:
        self._sharpen.set_value(v, silent=silent)

    def set_hdr(self, v: float, silent: bool = True) -> None:
        self._hdr.set_value(v, silent=silent)

    def set_brightness(self, v: float, silent: bool = True) -> None:
        self._brightness.set_value(v, silent=silent)

    def set_contrast(self, v: float, silent: bool = True) -> None:
        self._contrast.set_value(v, silent=silent)

    def set_grayscale(self, v: bool, silent: bool = True) -> None:
        self._cb_bw.blockSignals(silent)
        self._cb_bw.setChecked(v)
        self._cb_bw.blockSignals(False)
        if not silent:
            self._emit()

    def set_shadow_highlight(self, v: float, silent: bool = True) -> None:
        self._shadow_highlight.set_value(v, silent=silent)
