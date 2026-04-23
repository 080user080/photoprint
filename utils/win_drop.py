"""
Windows Drag & Drop через WM_DROPFILES.
Обходить обмеження PyQt6 6.7+ на Windows 10/11.
Підключається до MainWindow через installNativeEventFilter.
"""
import struct
import ctypes
import ctypes.wintypes
from PyQt6.QtCore import QAbstractNativeEventFilter

WM_DROPFILES = 0x0233

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
    count = _shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
    files = []
    for i in range(count):
        buf = ctypes.create_unicode_buffer(260)
        _shell32.DragQueryFileW(hdrop, i, buf, 260)
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
            raw  = (ctypes.c_uint8 * 32).from_address(addr)
            data = bytes(raw)
            # MSG layout (64-bit): HWND(8) + UINT(4) + pad(4) + WPARAM(8) + LPARAM(8)
            msg_id = struct.unpack_from('<I', data, 8)[0]
            if msg_id == WM_DROPFILES:
                wparam = struct.unpack_from('<Q', data, 16)[0]
                files  = _read_drop_files(wparam)
                if files:
                    self._cb(files)
                return True, 0
        except Exception:
            pass
        return False, 0
