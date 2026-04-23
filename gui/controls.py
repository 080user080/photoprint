"""
Панель інструментів: яскравість, контраст, різкість, HDR, Ч/Б, перспектива.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class _SliderRow(QWidget):
    changed = pyqtSignal(float)
    auto_clicked = pyqtSignal()

    def __init__(self, label, min_val, max_val, default, show_auto=False, parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._scale = 100

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(72)
        lbl.setStyleSheet("color: #111111; font-size: 12px;")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val * self._scale), int(max_val * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_change)

        self._val_lbl = QLabel(f"{default:.2f}")
        self._val_lbl.setFixedWidth(38)
        self._val_lbl.setStyleSheet("color: #111111; font-size: 12px;")

        layout.addWidget(lbl)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._val_lbl)

        if show_auto:
            btn = QPushButton("Авто")
            btn.setFixedWidth(44)
            btn.setFixedHeight(22)
            btn.setStyleSheet(
                "background:#4A7FC1; color:white; font-size:11px;"
                "border-radius:3px; padding:0px;"
            )
            btn.clicked.connect(self.auto_clicked)
            layout.addWidget(btn)

    def _on_change(self, raw):
        v = raw / self._scale
        self._val_lbl.setText(f"{v:.2f}")
        self.changed.emit(v)

    def value(self):
        return self._slider.value() / self._scale

    def set_value(self, v, silent=False):
        if silent:
            self._slider.blockSignals(True)
        self._slider.setValue(int(v * self._scale))
        self._val_lbl.setText(f"{v:.2f}")
        if silent:
            self._slider.blockSignals(False)

    def reset(self):
        self.set_value(0.0, silent=True)


class ControlsPanel(QWidget):
    changed = pyqtSignal(dict)
    auto_brightness_clicked   = pyqtSignal()
    auto_contrast_clicked     = pyqtSignal()
    auto_sharpen_clicked      = pyqtSignal()
    perspective_auto_clicked  = pyqtSignal()
    perspective_manual_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _group_style(self):
        return (
            "QGroupBox { border:1px solid #BBBBBB; border-radius:4px; "
            "margin-top:6px; background:#FFFFFF; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; "
            "padding:0 4px; color:#1F3864; font-weight:bold; font-size:12px; }"
        )

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Яскравість
        br_box = QGroupBox("Яскравість")
        br_box.setStyleSheet(self._group_style())
        br_lay = QVBoxLayout(br_box)
        self._brightness = _SliderRow("Яскравість", -1.0, 1.0, 0.0, show_auto=True)
        self._brightness.changed.connect(self._emit)
        self._brightness.auto_clicked.connect(self.auto_brightness_clicked)
        br_lay.addWidget(self._brightness)

        # Контраст
        ct_box = QGroupBox("Контраст")
        ct_box.setStyleSheet(self._group_style())
        ct_lay = QVBoxLayout(ct_box)
        self._contrast = _SliderRow("Контраст", -1.0, 1.0, 0.0, show_auto=True)
        self._contrast.changed.connect(self._emit)
        self._contrast.auto_clicked.connect(self.auto_contrast_clicked)
        ct_lay.addWidget(self._contrast)

        # Різкість
        sh_box = QGroupBox("Різкість")
        sh_box.setStyleSheet(self._group_style())
        sh_lay = QVBoxLayout(sh_box)
        self._sharpen = _SliderRow("Різкість", 0.0, 1.0, 0.4, show_auto=True)
        self._sharpen.changed.connect(self._emit)
        self._sharpen.auto_clicked.connect(self.auto_sharpen_clicked)
        sh_lay.addWidget(self._sharpen)

        # HDR
        hdr_box = QGroupBox("HDR")
        hdr_box.setStyleSheet(self._group_style())
        hdr_lay = QVBoxLayout(hdr_box)
        self._hdr = _SliderRow("HDR", 0.0, 1.0, 0.0)
        self._hdr.changed.connect(self._emit)
        hdr_lay.addWidget(self._hdr)

        # Чорно-білий
        bw_box = QGroupBox("Режим")
        bw_box.setStyleSheet(self._group_style())
        bw_lay = QVBoxLayout(bw_box)
        self._cb_bw = QCheckBox("Чорно-білий")
        self._cb_bw.setStyleSheet("color:#111111; font-size:12px;")
        self._cb_bw.toggled.connect(self._emit)
        bw_lay.addWidget(self._cb_bw)

        # Перспектива
        persp_box = QGroupBox("Перспектива")
        persp_box.setStyleSheet(self._group_style())
        persp_lay = QHBoxLayout(persp_box)
        btn_pa = QPushButton("Авто")
        btn_pm = QPushButton("Ручна (4 точки)")
        btn_pa.clicked.connect(self.perspective_auto_clicked)
        btn_pm.clicked.connect(self.perspective_manual_clicked)
        persp_lay.addWidget(btn_pa)
        persp_lay.addWidget(btn_pm)

        # Скинути
        btn_reset = QPushButton("Скинути всі")
        btn_reset.setStyleSheet(
            "background:#888888; color:white; border-radius:4px; padding:5px;"
        )
        btn_reset.clicked.connect(self.reset_all)

        root.addWidget(br_box)
        root.addWidget(ct_box)
        root.addWidget(sh_box)
        root.addWidget(hdr_box)
        root.addWidget(bw_box)
        root.addWidget(persp_box)
        root.addWidget(btn_reset)
        root.addStretch()

    def _emit(self, *_):
        self.changed.emit(self.values())

    def values(self):
        return {
            "brightness":       self._brightness.value(),
            "contrast":         self._contrast.value(),
            "sharpen_strength": self._sharpen.value(),
            "hdr_strength":     self._hdr.value(),
            "grayscale":        self._cb_bw.isChecked(),
        }

    def reset_all(self):
        for w in (self._brightness, self._contrast, self._sharpen, self._hdr):
            w.reset()
        self._cb_bw.setChecked(False)
        self._emit()

    def set_sharpen(self, v): self._sharpen.set_value(v, silent=True)
    def set_hdr(self, v):     self._hdr.set_value(v, silent=True)

    def set_brightness(self, v, silent=True):
        self._brightness.set_value(v, silent=silent)

    def set_contrast(self, v, silent=True):
        self._contrast.set_value(v, silent=silent)
