"""
Drag & Drop через WM_DROPFILES — фінальна версія з правильними типами.
"""
import sys
import ctypes
import ctypes.wintypes
import struct
from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, QAbstractNativeEventFilter

WM_DROPFILES = 0x0233

shell32 = ctypes.windll.shell32
shell32.DragQueryFileW.restype  = ctypes.c_uint
shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint]
shell32.DragFinish.restype      = None
shell32.DragFinish.argtypes     = [ctypes.c_void_p]
shell32.DragAcceptFiles.restype = None
shell32.DragAcceptFiles.argtypes= [ctypes.c_void_p, ctypes.c_bool]


def setup_drop(hwnd_int):
    shell32.DragAcceptFiles(hwnd_int, True)
    print(f"[DROP] DragAcceptFiles OK, hwnd={hwnd_int}")


def query_drop_files(hdrop_int):
    hdrop = ctypes.c_void_p(hdrop_int)
    count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
    print(f"[QUERY] файлів: {count}")
    files = []
    for i in range(count):
        buf = ctypes.create_unicode_buffer(260)
        shell32.DragQueryFileW(hdrop, i, buf, 260)
        files.append(buf.value)
    shell32.DragFinish(hdrop)
    return files


class WinEventFilter(QAbstractNativeEventFilter):
    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def nativeEventFilter(self, event_type, message):
        try:
            addr = int(message)
            raw  = (ctypes.c_uint8 * 32).from_address(addr)
            data = bytes(raw)
            msg_id = struct.unpack_from('<I', data, 8)[0]

            if msg_id == WM_DROPFILES:
                wparam = struct.unpack_from('<Q', data, 16)[0]
                print(f"[WM_DROPFILES] wparam={wparam}")
                files = query_drop_files(wparam)
                for f in files:
                    print(f"  → {f}")
                self._cb(files)
                return True, 0
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
        return False, 0


class DropList(QListWidget):
    def add_paths(self, paths):
        for p in paths:
            self.addItem(QListWidgetItem(p))


class DropWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WM_DROPFILES Test v4")
        self.resize(500, 400)
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        self._lbl = QLabel("Ініціалізація...")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("font-size:14px; padding:10px;")
        self.drop_list = DropList()
        lay.addWidget(self._lbl)
        lay.addWidget(self.drop_list)
        QTimer.singleShot(300, self._setup)

    def _setup(self):
        hwnd_int = int(self.winId())
        print(f"[HWND] {hwnd_int}")
        setup_drop(hwnd_int)
        self._lbl.setText("Перетягни файл із провідника на це вікно")
        self._lbl.setStyleSheet("font-size:14px; color:#006600; padding:10px;")
        print("[READY] Готово")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DropWindow()
    ev_filter = WinEventFilter(lambda files: w.drop_list.add_paths(files))
    app.installNativeEventFilter(ev_filter)
    w.show()
    print(f"[INFO] PyQt6: {__import__('PyQt6.QtCore', fromlist=['PYQT_VERSION_STR']).PYQT_VERSION_STR}")
    sys.exit(app.exec())
