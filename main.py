"""
Точка входу програми PhotoPrint.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from gui.main_window import MainWindow


STYLE = """
/* ─── Базовий фон і текст ─── */
QMainWindow, QWidget {
    background-color: #F0F2F5;
    color: #111111;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

/* ─── Кнопки ─── */
QPushButton {
    background-color: #2E5FA3;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover   { background-color: #3A70BB; }
QPushButton:pressed { background-color: #1F4080; }
QPushButton:disabled { background-color: #AAAAAA; color: #DDDDDD; }

/* ─── GroupBox ─── */
QGroupBox {
    border: 1px solid #C0C4CC;
    border-radius: 4px;
    margin-top: 8px;
    background: #FFFFFF;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #1F3864;
    font-weight: bold;
}

/* ─── Список черги ─── */
QListWidget {
    border: 1px solid #C0C4CC;
    border-radius: 4px;
    background: #FFFFFF;
    color: #111111;
}
QListWidget::item { color: #111111; padding: 2px 4px; }
QListWidget::item:selected {
    background: #D6EAF8;
    color: #111111;
}

/* ─── Мітки ─── */
QLabel { color: #111111; }

/* ─── RadioButton / CheckBox ─── */
QRadioButton { color: #111111; }
QCheckBox    { color: #111111; }

/* ─── Слайдер ─── */
QSlider::groove:horizontal {
    height: 4px;
    background: #C8CDD6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #2E5FA3;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #2E5FA3;
    border-radius: 2px;
}

/* ─── ProgressBar ─── */
QProgressBar {
    border: 1px solid #C0C4CC;
    border-radius: 4px;
    text-align: center;
    color: #111111;
    background: #FFFFFF;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #2E5FA3;
    border-radius: 3px;
}

/* ─── ScrollBar ─── */
QScrollBar:vertical {
    width: 8px;
    background: #E8EAF0;
}
QScrollBar::handle:vertical {
    background: #AAAAAA;
    border-radius: 4px;
    min-height: 20px;
}

/* ─── SpinBox / LineEdit ─── */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #FFFFFF;
    color: #111111;
    border: 1px solid #C0C4CC;
    border-radius: 3px;
    padding: 2px 4px;
}

/* ─── ScrollArea ─── */
QScrollArea { border: none; background: transparent; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PhotoPrint")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
