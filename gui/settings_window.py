"""
Вікно налаштувань — відображає і редагує settings.ini.
Статичне (не модальне): можна тримати відкритим поруч із головним вікном.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal
from config import app_settings


class SettingsWindow(QWidget):
    """Вікно налаштувань. Зміни набирають силу після натискання Зберегти."""

    settings_saved = pyqtSignal(dict)   # нові налаштування

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Налаштування")
        self.setMinimumWidth(420)
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
        root.setSpacing(12)

        # === Обробка ===
        proc_box = QGroupBox("Обробка")
        proc_form = QFormLayout(proc_box)

        self._cb_autofix    = QCheckBox()
        self._cb_hdr        = QCheckBox()
        self._cb_perspective = QCheckBox()
        proc_form.addRow("Auto Fix за замовчуванням:", self._cb_autofix)
        proc_form.addRow("HDR в Auto Fix:",            self._cb_hdr)
        proc_form.addRow("Авто-перспектива:",          self._cb_perspective)

        self._spin_sharpen = QDoubleSpinBox()
        self._spin_sharpen.setRange(0.0, 1.0)
        self._spin_sharpen.setSingleStep(0.05)
        self._spin_sharpen.setDecimals(2)

        self._spin_hdr = QDoubleSpinBox()
        self._spin_hdr.setRange(0.0, 1.0)
        self._spin_hdr.setSingleStep(0.05)
        self._spin_hdr.setDecimals(2)

        proc_form.addRow("Сила різкості (0–1):", self._spin_sharpen)
        proc_form.addRow("Сила HDR (0–1):",      self._spin_hdr)

        # === Вихід ===
        out_box = QGroupBox("Збереження")
        out_form = QFormLayout(out_box)

        self._spin_quality = QSpinBox()
        self._spin_quality.setRange(50, 100)

        self._edit_folder = QLineEdit()
        self._edit_folder.setPlaceholderText("(порожньо = не зберігати)")
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_folder)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self._edit_folder)
        folder_row.addWidget(btn_browse)

        out_form.addRow("Якість JPG (50–100):", self._spin_quality)
        out_form.addRow("Папка збереження:",    folder_row)

        # === Принтер ===
        print_box = QGroupBox("Принтер")
        print_form = QFormLayout(print_box)

        self._edit_printer = QLineEdit()
        self._edit_printer.setPlaceholderText("priPrinter")
        print_form.addRow("Назва принтера:", self._edit_printer)

        # === Режим за замовчуванням ===
        mode_box = QGroupBox("Режим запуску")
        mode_form = QFormLayout(mode_box)
        self._cb_default_auto = QCheckBox("Авто (інакше — Ручний)")
        mode_form.addRow(self._cb_default_auto)

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

        root.addWidget(proc_box)
        root.addWidget(out_box)
        root.addWidget(print_box)
        root.addWidget(mode_box)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Завантаження / збереження
    # ------------------------------------------------------------------

    def load_from_file(self):
        s = app_settings.load()
        self._apply_settings(s)

    def _apply_settings(self, s: dict):
        self._cb_autofix.setChecked(s.get("autofix_enabled", True))
        self._cb_hdr.setChecked(s.get("hdr_in_autofix", True))
        self._cb_perspective.setChecked(s.get("auto_perspective", True))
        self._spin_sharpen.setValue(s.get("sharpen_strength", 0.4))
        self._spin_hdr.setValue(s.get("hdr_strength", 0.5))
        self._spin_quality.setValue(s.get("jpg_quality", 95))
        self._edit_folder.setText(s.get("save_folder", ""))
        self._edit_printer.setText(s.get("printer_name", "priPrinter"))
        self._cb_default_auto.setChecked(s.get("default_mode", "auto") == "auto")

    def _collect_settings(self) -> dict:
        return {
            "autofix_enabled":   self._cb_autofix.isChecked(),
            "hdr_in_autofix":    self._cb_hdr.isChecked(),
            "auto_perspective":  self._cb_perspective.isChecked(),
            "sharpen_strength":  self._spin_sharpen.value(),
            "hdr_strength":      self._spin_hdr.value(),
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
