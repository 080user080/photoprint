"""
Мінімальний тест Drag & Drop з Windows OLE fix.
PyQt6 6.7+ на Windows 10/11 потребує явної реєстрації OLE drop target.
"""
import sys
import ctypes
import ctypes.wintypes
from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QEvent, QTimer
from PyQt6.QtGui import QColor


def enable_ole_drag_drop(hwnd):
    """
    Реєструє вікно як OLE DragDrop target.
    Без цього Windows 10/11 + PyQt6 6.7+ блокує drag із провідника.
    """
    try:
        ole32 = ctypes.windll.ole32
        ole32.OleInitialize(None)
        # DragAcceptFiles через shell32
        ctypes.windll.shell32.DragAcceptFiles(hwnd, True)
        print(f"[OLE] DragAcceptFiles встановлено для hwnd={hwnd}")
        return True
    except Exception as e:
        print(f"[OLE] Помилка: {e}")
        return False


class TestList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().installEventFilter(self)
        self.setDropIndicatorShown(True)
        print("[INIT] TestList створено")

    def eventFilter(self, source, event):
        t = event.type()
        if source is self.viewport():
            if t == QEvent.Type.DragEnter:
                print("[EVENTFILTER] DragEnter на viewport")
                self.dragEnterEvent(event)
                return True
            elif t == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
                return True
            elif t == QEvent.Type.Drop:
                print("[EVENTFILTER] Drop на viewport")
                self.dropEvent(event)
                return True
        return super().eventFilter(source, event)

    def dragEnterEvent(self, event):
        print(f"[dragEnterEvent] hasUrls={event.mimeData().hasUrls()}")
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        print(f"[dropEvent] файлів: {len(event.mimeData().urls())}")
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            print(f"  → {path}")
            self.addItem(QListWidgetItem(path))
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DRAG DROP TEST v2 (OLE fix)")
        self.resize(500, 400)

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        self._lbl = QLabel("Перетягни файл сюди ↓\nЗастосовується OLE fix...")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("font-size:14px; color:#333; padding:10px;")

        self.list = TestList()
        lay.addWidget(self._lbl)
        lay.addWidget(self.list)

        # Застосовуємо OLE fix після показу вікна (hwnd доступний тільки після show)
        QTimer.singleShot(100, self._apply_ole_fix)

    def _apply_ole_fix(self):
        hwnd = int(self.winId())
        print(f"[OLE] Застосовую fix для hwnd={hwnd}")
        ok = enable_ole_drag_drop(hwnd)
        if ok:
            self._lbl.setText("OLE fix застосовано ✓\nПеретягни файл на список нижче")
            self._lbl.setStyleSheet("font-size:14px; color:#006600; padding:10px;")
        else:
            self._lbl.setText("OLE fix НЕ спрацював\nДив. консоль")
            self._lbl.setStyleSheet("font-size:14px; color:#cc0000; padding:10px;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestWindow()
    w.show()
    print(f"[INFO] PyQt6: {__import__('PyQt6.QtCore', fromlist=['PYQT_VERSION_STR']).PYQT_VERSION_STR}")
    sys.exit(app.exec())
