"""
Windows Drag & Drop через WM_DROPFILES.
Обходить обмеження PyQt6 6.7+ на Windows 10/11.
Підключається до MainWindow через installNativeEventFilter.
"""
import struct
import ctypes
import ctypes.wintypes
from PyQt6.QtCore import QAbstractNativeEventFilter

# Константи Windows API
WM_DROPFILES = 0x0233
UINT_MAX = 0xFFFFFFFF
MAX_PATH = 260

# Константи для структури MSG (64-bit)
MSG_SIZE = 32
MSG_HWND_OFFSET = 0
MSG_ID_OFFSET = 8
MSG_WPARAM_OFFSET = 16

# --- shell32 з правильними типами ---
_shell32 = ctypes.windll.shell32
_shell32.DragQueryFileW.restype   = ctypes.c_uint
_shell32.DragQueryFileW.argtypes  = [ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_wchar_p, ctypes.c_uint]
_shell32.DragFinish.restype       = None
_shell32.DragFinish.argtypes      = [ctypes.c_void_p]
_shell32.DragAcceptFiles.restype  = None
_shell32.DragAcceptFiles.argtypes = [ctypes.c_void_p, ctypes.c_bool]


def register_drop_window(hwnd: int):
    """Реєструє вікно для прийому WM_DROPFILES."""
    _shell32.DragAcceptFiles(hwnd, True)


def _read_drop_files(hdrop_int: int) -> list[str]:
    hdrop = ctypes.c_void_p(hdrop_int)
    count = _shell32.DragQueryFileW(hdrop, UINT_MAX, None, 0)
    files = []
    for i in range(count):
        buf = ctypes.create_unicode_buffer(MAX_PATH)
        _shell32.DragQueryFileW(hdrop, i, buf, MAX_PATH)
        files.append(buf.value)
    _shell32.DragFinish(hdrop)
    return files


class DropEventFilter(QAbstractNativeEventFilter):
    """
    Глобальний фільтр подій застосунку.
    При отриманні WM_DROPFILES викликає callback(list[str]).
    """

    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def nativeEventFilter(self, event_type, message):
        try:
            addr = int(message)
            raw  = (ctypes.c_uint8 * MSG_SIZE).from_address(addr)
            data = bytes(raw)
            # MSG layout (64-bit): HWND(8) + UINT(4) + pad(4) + WPARAM(8) + LPARAM(8)
            msg_id = struct.unpack_from('<I', data, MSG_ID_OFFSET)[0]
            if msg_id == WM_DROPFILES:
                wparam = struct.unpack_from('<Q', data, MSG_WPARAM_OFFSET)[0]
                files  = _read_drop_files(wparam)
                if files:
                    self._cb(files)
                return True, 0
        except Exception:
            pass
        return False, 0
