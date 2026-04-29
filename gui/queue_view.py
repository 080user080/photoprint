"""
Список файлів у черзі.
Drag & Drop сумісний з Windows 10/11.
"""

from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QListWidget, QListWidgetItem, QAbstractItemView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QUrl
from PyQt6.QtGui  import QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent
import os

# Константи для розмірів черги
QUEUE_MIN_WIDTH = 150
QUEUE_MAX_WIDTH = 500

# Константи для кольорів статусів
COLOR_PENDING = QColor("#FFFFFF")
COLOR_CURRENT = QColor("#D6EAF8")
COLOR_DONE = QColor("#D5F5E3")
COLOR_ERROR = QColor("#FADBD8")
COLOR_SKIPPED = QColor("#F0F0F0")
COLOR_TEXT = QColor("#111111")

# Константи для префіксів статусів
PREFIX_DONE = "✓ "
PREFIX_ERROR = "✗ "
PREFIX_SKIPPED = "– "
PREFIX_CURRENT = "▶ "


class QueueView(QListWidget):
    files_dropped     = pyqtSignal(list)   # list[str]
    selection_changed = pyqtSignal(str)

    _COLOR = {
        "pending": COLOR_PENDING,
        "current": COLOR_CURRENT,
        "done":    COLOR_DONE,
        "error":   COLOR_ERROR,
        "skipped": COLOR_SKIPPED,
    }
    _PREFIX = {"done": PREFIX_DONE, "error": PREFIX_ERROR, "skipped": PREFIX_SKIPPED, "current": PREFIX_CURRENT}

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # --- Критично для Windows ---
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setDropIndicatorShown(True)
        # ----------------------------

        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setMinimumWidth(QUEUE_MIN_WIDTH)
        self.setMaximumWidth(QUEUE_MAX_WIDTH)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.itemClicked.connect(self._on_clicked)

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def set_files(self, paths: List[str]) -> None:
        self.clear()
        for p in paths:
            self._add_item(p)

    def add_files(self, paths: List[str]) -> None:
        existing = set(self._all_paths())
        for p in paths:
            if p not in existing:
                self._add_item(p)
                existing.add(p)

    def mark_current(self, idx: int) -> None:  self._set_status(idx, "current")
    def mark_done(self, idx: int) -> None:     self._set_status(idx, "done")
    def mark_error(self, idx: int) -> None:    self._set_status(idx, "error")
    def mark_skipped(self, idx: int) -> None:  self._set_status(idx, "skipped")

    def get_all_paths(self) -> List[str]:
        return self._all_paths()

    def get_path(self, idx: int) -> Optional[str]:
        item = self.item(idx)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def clear_queue(self) -> None:
        self.clear()

    # ------------------------------------------------------------------
    # Drag & Drop — перевизначаємо на viewport теж (Windows)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = self._urls_to_paths(event.mimeData().urls())
        if paths:
            self.add_files(paths)
            self.files_dropped.emit(paths)
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()

    # ------------------------------------------------------------------
    # Внутрішнє
    # ------------------------------------------------------------------

    def _urls_to_paths(self, urls: List[QUrl]) -> List[str]:
        result = []
        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path):
                result.append(path)
            elif os.path.isdir(path):
                from utils.file_utils import collect_images_from_folder
                result.extend(collect_images_from_folder(path))
        return result

    def _add_item(self, path: str, status: str = "pending") -> None:
        name = os.path.basename(path)
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setBackground(self._COLOR["pending"])
        item.setForeground(COLOR_TEXT)
        item.setToolTip(path)
        self.addItem(item)

    def _set_status(self, idx: int, status: str) -> None:
        item = self.item(idx)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        name = os.path.basename(path) if path else ""
        prefix = self._PREFIX.get(status, "")
        item.setText(prefix + name)
        item.setBackground(self._COLOR.get(status, self._COLOR["pending"]))
        item.setForeground(COLOR_TEXT)
        if status == "current":
            self.setCurrentRow(idx)
            self.scrollToItem(item)

    def _all_paths(self) -> List[str]:
        result = []
        for i in range(self.count()):
            item = self.item(i)
            if item:
                p = item.data(Qt.ItemDataRole.UserRole)
                if p:
                    result.append(p)
        return result

    def _on_clicked(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        if p:
            self.selection_changed.emit(p)
