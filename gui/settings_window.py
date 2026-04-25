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

        self._cb_autofix        = QCheckBox()
        self._cb_auto_apply     = QCheckBox()
        self._cb_hdr            = QCheckBox()
        self._cb_perspective    = QCheckBox()
        proc_form.addRow("Auto Fix за замовчуванням:",    self._cb_autofix)
        proc_form.addRow("Авто-застосувати при завантаженні:", self._cb_auto_apply)
        proc_form.addRow("HDR в Auto Fix:",                self._cb_hdr)
        proc_form.addRow("Авто-перспектива:",              self._cb_perspective)

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

        # === Класифікація документів ===
        cls_box = QGroupBox("Класифікація документів")
        cls_form = QFormLayout(cls_box)

        self._spin_bw_std = QDoubleSpinBox()
        self._spin_bw_std.setRange(1.0, 100.0)
        self._spin_bw_std.setSingleStep(1.0)
        self._spin_bw_std.setDecimals(1)

        self._spin_edge_ratio = QDoubleSpinBox()
        self._spin_edge_ratio.setRange(0.001, 0.5)
        self._spin_edge_ratio.setSingleStep(0.01)
        self._spin_edge_ratio.setDecimals(3)

        self._spin_line_count = QSpinBox()
        self._spin_line_count.setRange(0, 50)

        cls_form.addRow("Поріг std(a,b) для ЧБ:",    self._spin_bw_std)
        cls_form.addRow("Мін. частка країв (0–1):",   self._spin_edge_ratio)
        cls_form.addRow("Мін. кількість ліній:",      self._spin_line_count)

        # === Авто-різкість ===
        sh_box = QGroupBox("Авто-різкість")
        sh_form = QFormLayout(sh_box)

        self._spin_asharp_thresh = QDoubleSpinBox()
        self._spin_asharp_thresh.setRange(1.0, 500.0)
        self._spin_asharp_thresh.setSingleStep(5.0)
        self._spin_asharp_thresh.setDecimals(1)

        self._spin_asharp_max = QDoubleSpinBox()
        self._spin_asharp_max.setRange(0.1, 1.0)
        self._spin_asharp_max.setSingleStep(0.05)
        self._spin_asharp_max.setDecimals(2)

        sh_form.addRow("Поріг Laplacian variance:",  self._spin_asharp_thresh)
        sh_form.addRow("Макс. сила різкості (0–1):", self._spin_asharp_max)

        # === Авто-яскравість/контраст ===
        pct_box = QGroupBox("Авто-яскравість/контраст")
        pct_form = QFormLayout(pct_box)

        self._spin_pct_low = QDoubleSpinBox()
        self._spin_pct_low.setRange(0.0, 25.0)
        self._spin_pct_low.setSingleStep(1.0)
        self._spin_pct_low.setDecimals(1)

        self._spin_pct_high = QDoubleSpinBox()
        self._spin_pct_high.setRange(75.0, 100.0)
        self._spin_pct_high.setSingleStep(1.0)
        self._spin_pct_high.setDecimals(1)

        pct_form.addRow("Нижній процентиль (%):",  self._spin_pct_low)
        pct_form.addRow("Верхній процентиль (%):", self._spin_pct_high)

        # === Чорно-білий ===
        bw_box = QGroupBox("Чорно-білий документ")
        bw_form = QFormLayout(bw_box)

        self._cb_bw_binary = QCheckBox("Адаптивна бінаризація")
        bw_form.addRow(self._cb_bw_binary)

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
        root.addWidget(cls_box)
        root.addWidget(sh_box)
        root.addWidget(pct_box)
        root.addWidget(bw_box)
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
        self._cb_auto_apply.setChecked(s.get("auto_apply_autofix", True))
        self._cb_hdr.setChecked(s.get("hdr_in_autofix", True))
        self._cb_perspective.setChecked(s.get("auto_perspective", True))
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
