"""
Точка входу програми PhotoPrint.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from gui.main_window import MainWindow

# Константи для кольорів
COLOR_BG = "#F0F2F5"
COLOR_TEXT = "#111111"
COLOR_TEXT_DISABLED = "#DDDDDD"
COLOR_BUTTON = "#2E5FA3"
COLOR_BUTTON_HOVER = "#3A70BB"
COLOR_BUTTON_PRESSED = "#1F4080"
COLOR_BUTTON_DISABLED = "#AAAAAA"
COLOR_BORDER = "#C0C4CC"
COLOR_GROUPBOX_TITLE = "#1F3864"
COLOR_LIST_SELECTED = "#D6EAF8"
COLOR_SLIDER_GROOVE = "#C8CDD6"
COLOR_SLIDER_HANDLE = "#2E5FA3"
COLOR_SCROLLBAR_BG = "#E8EAF0"
COLOR_SCROLLBAR_HANDLE = "#AAAAAA"
COLOR_WHITE = "#FFFFFF"

# Константи для розмірів та відступів
FONT_SIZE = 13
BORDER_RADIUS = 4
BORDER_RADIUS_SMALL = 3
BORDER_RADIUS_SMALLER = 2
BORDER_RADIUS_SLIDER = 7
BUTTON_PADDING = "6px 12px"
GROUPBOX_MARGIN_TOP = "8px"
GROUPBOX_TITLE_LEFT = "8px"
GROUPBOX_TITLE_PADDING = "0 4px"
LIST_ITEM_PADDING = "2px 4px"
SLIDER_GROOVE_HEIGHT = "4px"
SLIDER_HANDLE_SIZE = "14px"
SLIDER_HANDLE_MARGIN = "-5px 0"
PROGRESSBAR_HEIGHT = "18px"
SCROLLBAR_WIDTH = "8px"
SCROLLBAR_HANDLE_MIN_HEIGHT = "20px"
SPINBOX_PADDING = "2px 4px"

STYLE = f"""
/* ─── Базовий фон і текст ─── */
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: {FONT_SIZE}px;
}}

/* ─── Кнопки ─── */
QPushButton {{
    background-color: {COLOR_BUTTON};
    color: {COLOR_WHITE};
    border: none;
    border-radius: {BORDER_RADIUS}px;
    padding: {BUTTON_PADDING};
}}
QPushButton:hover   {{ background-color: {COLOR_BUTTON_HOVER}; }}
QPushButton:pressed {{ background-color: {COLOR_BUTTON_PRESSED}; }}
QPushButton:disabled {{ background-color: {COLOR_BUTTON_DISABLED}; color: {COLOR_TEXT_DISABLED}; }}

/* ─── GroupBox ─── */
QGroupBox {{
    border: 1px solid {COLOR_BORDER};
    border-radius: {BORDER_RADIUS}px;
    margin-top: {GROUPBOX_MARGIN_TOP};
    background: {COLOR_WHITE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {GROUPBOX_TITLE_LEFT};
    padding: {GROUPBOX_TITLE_PADDING};
    color: {COLOR_GROUPBOX_TITLE};
    font-weight: bold;
}}

/* ─── Список черги ─── */
QListWidget {{
    border: 1px solid {COLOR_BORDER};
    border-radius: {BORDER_RADIUS}px;
    background: {COLOR_WHITE};
    color: {COLOR_TEXT};
}}
QListWidget::item {{ color: {COLOR_TEXT}; padding: {LIST_ITEM_PADDING}; }}
QListWidget::item:selected {{
    background: {COLOR_LIST_SELECTED};
    color: {COLOR_TEXT};
}}

/* ─── Мітки ─── */
QLabel {{ color: {COLOR_TEXT}; }}

/* ─── RadioButton / CheckBox ─── */
QRadioButton {{ color: {COLOR_TEXT}; }}
QCheckBox    {{ color: {COLOR_TEXT}; }}

/* ─── Слайдер ─── */
QSlider::groove:horizontal {{
    height: {SLIDER_GROOVE_HEIGHT};
    background: {COLOR_SLIDER_GROOVE};
    border-radius: {BORDER_RADIUS_SMALLER}px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_SLIDER_HANDLE};
    width: {SLIDER_HANDLE_SIZE};
    height: {SLIDER_HANDLE_SIZE};
    margin: {SLIDER_HANDLE_MARGIN};
    border-radius: {BORDER_RADIUS_SLIDER}px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_SLIDER_HANDLE};
    border-radius: {BORDER_RADIUS_SMALLER}px;
}}

/* ─── ProgressBar ─── */
QProgressBar {{
    border: 1px solid {COLOR_BORDER};
    border-radius: {BORDER_RADIUS}px;
    text-align: center;
    color: {COLOR_TEXT};
    background: {COLOR_WHITE};
    height: {PROGRESSBAR_HEIGHT};
}}
QProgressBar::chunk {{
    background-color: {COLOR_BUTTON};
    border-radius: {BORDER_RADIUS_SMALL}px;
}}

/* ─── ScrollBar ─── */
QScrollBar:vertical {{
    width: {SCROLLBAR_WIDTH};
    background: {COLOR_SCROLLBAR_BG};
}}
QScrollBar::handle:vertical {{
    background: {COLOR_SCROLLBAR_HANDLE};
    border-radius: {BORDER_RADIUS_SMALL}px;
    min-height: {SCROLLBAR_HANDLE_MIN_HEIGHT};
}}

/* ─── SpinBox / LineEdit ─── */
QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {COLOR_WHITE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: {BORDER_RADIUS_SMALL}px;
    padding: {SPINBOX_PADDING};
}}

/* ─── ScrollArea ─── */
QScrollArea {{ border: none; background: transparent; }}
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
