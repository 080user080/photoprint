"""
Мінімальний тест Drag & Drop — запусти і спробуй перетягнути файл.
Покаже точно де проблема.
"""
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QAbstractItemView, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QColor

class TestList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().installEventFilter(self)
        self.setDropIndicatorShown(True)
        print("[INIT] TestList створено, eventFilter встановлено на viewport")

    def eventFilter(self, source, event):
        t = event.type()
        if t in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop, QEvent.Type.DragLeave):
            print(f"[EVENTFILTER] source={source.__class__.__name__} event={t}")
        if source is self.viewport():
            if t == QEvent.Type.DragEnter:
                print("[EVENTFILTER] → DragEnter на viewport")
                self.dragEnterEvent(event)
                return True
            elif t == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
                return True
            elif t == QEvent.Type.Drop:
                print("[EVENTFILTER] → Drop на viewport")
                self.dropEvent(event)
                return True
        return super().eventFilter(source, event)

    def dragEnterEvent(self, event):
        print(f"[dragEnterEvent] hasUrls={event.mimeData().hasUrls()}")
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            print("[dragEnterEvent] ACCEPTED")
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        print(f"[dropEvent] hasUrls={event.mimeData().hasUrls()}")
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            print(f"[dropEvent] файл: {path}")
            self.addItem(QListWidgetItem(path))
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DRAG DROP TEST — дивись консоль")
        self.resize(500, 400)
        # НЕ викликаємо setAcceptDrops на MainWindow!

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        lbl = QLabel("Перетягни файл сюди ↓\nДив. консоль для діагностики")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size:16px; color:#333; padding:10px;")

        self.list = TestList()
        lay.addWidget(lbl)
        lay.addWidget(self.list)

        print("[WINDOW] MainWindow створено БЕЗ setAcceptDrops")
        print("[WINDOW] Тепер перетягни будь-який файл на список нижче")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestWindow()
    w.show()
    print(f"[INFO] PyQt6 версія: {__import__('PyQt6.QtCore', fromlist=['PYQT_VERSION_STR']).PYQT_VERSION_STR}")
    print(f"[INFO] Qt версія: {__import__('PyQt6.QtCore', fromlist=['QT_VERSION_STR']).QT_VERSION_STR}")
    sys.exit(app.exec())
