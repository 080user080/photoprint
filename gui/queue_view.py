"""
Список файлів у черзі.
Drag & Drop реалізовано через WM_DROPFILES у win_drop.py
"""

from PyQt6.QtWidgets import (
    QListWidget, QListWidgetItem, QAbstractItemView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui  import QColor
import os


class QueueView(QListWidget):
    files_dropped     = pyqtSignal(list)   # list[str]
    selection_changed = pyqtSignal(str)

    _COLOR = {
        "pending": QColor("#FFFFFF"),
        "current": QColor("#D6EAF8"),
        "done":    QColor("#D5F5E3"),
        "error":   QColor("#FADBD8"),
        "skipped": QColor("#F0F0F0"),
    }
    _PREFIX = {"done": "✓ ", "error": "✗ ", "skipped": "– ", "current": "▶ "}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.itemClicked.connect(self._on_clicked)

    def set_files(self, paths):
        self.clear()
        for p in paths:
            self._add_item(p)

    def add_files(self, paths):
        existing = set(self._all_paths())
        for p in paths:
            if p not in existing:
                self._add_item(p)
                existing.add(p)

    def mark_current(self, idx):  self._set_status(idx, "current")
    def mark_done(self, idx):     self._set_status(idx, "done")
    def mark_error(self, idx):    self._set_status(idx, "error")
    def mark_skipped(self, idx):  self._set_status(idx, "skipped")

    def get_all_paths(self):
        return self._all_paths()

    def get_path(self, idx):
        item = self.item(idx)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def clear_queue(self):
        self.clear()

    def _add_item(self, path, status="pending"):
        name = os.path.basename(path)
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setBackground(self._COLOR["pending"])
        item.setForeground(QColor("#111111"))
        item.setToolTip(path)
        self.addItem(item)

    def _set_status(self, idx, status):
        item = self.item(idx)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        name = os.path.basename(path) if path else ""
        prefix = self._PREFIX.get(status, "")
        item.setText(prefix + name)
        item.setBackground(self._COLOR.get(status, self._COLOR["pending"]))
        item.setForeground(QColor("#111111"))
        if status == "current":
            self.setCurrentRow(idx)
            self.scrollToItem(item)

    def _all_paths(self):
        result = []
        for i in range(self.count()):
            item = self.item(i)
            if item:
                p = item.data(Qt.ItemDataRole.UserRole)
                if p:
                    result.append(p)
        return result

    def _on_clicked(self, item):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p:
            self.selection_changed.emit(p)
