"""
Unit-тести для core/printer.py: _print_windows fallback ланцюжок (TODO 4.7).
"""

import os
import sys
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.printer import (
    _save_temp_jpg,
    _print_windows,
    _print_windows_win32,
    _print_unix,
    print_image,
)


# ============================================================
# Тест 1: _save_temp_jpg
# ============================================================

class TestSaveTempJpg:
    """Перевірка збереження тимчасового JPG."""

    def test_returns_path(self):
        """Повертає шлях до файлу."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        path = _save_temp_jpg(img)
        assert path.endswith(".jpg")
        assert os.path.basename(path).startswith("photoprint_")

    def test_file_exists(self):
        """Файл створюється на диску."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        path = _save_temp_jpg(img)
        assert os.path.exists(path)
        os.remove(path)

    def test_quality_param(self):
        """Якість передається в cv2.imencode."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        with patch("cv2.imencode") as mock_imencode:
            mock_imencode.return_value = (True, np.frombuffer(b"test", dtype=np.uint8))
            _save_temp_jpg(img, quality=80)
            mock_imencode.assert_called_once()
            args = mock_imencode.call_args[0]
            # args[0] = суфікс, args[1] = image, args[2] = params
            assert args[2][0] == cv2.IMWRITE_JPEG_QUALITY
            assert args[2][1] == 80


# ============================================================
# Тест 2: _print_windows_win32 — ImportError коли pywin32 немає
# ============================================================

class TestPrintWindowsWin32:
    """Перевірка _print_windows_win32."""

    def test_import_error_when_pywin32_missing(self):
        """Коли pywin32 не встановлено — кидає ImportError."""
        with patch.dict("sys.modules", {"win32print": None, "win32ui": None}):
            with pytest.raises(ImportError):
                _print_windows_win32("test.jpg", "Printer")


# ============================================================
# Тест 3: _print_windows — fallback ланцюжок
# ============================================================

class TestPrintWindowsFallback:
    """Перевірка ланцюжка fallback у _print_windows."""

    def test_win32print_success(self):
        """Якщо win32print доступний — використовується він."""
        with patch("core.printer._print_windows_win32") as mock_win32:
            _print_windows("test.jpg", "Printer")
            mock_win32.assert_called_once_with("test.jpg", "Printer")

    def test_import_error_falls_back_to_mspaint(self):
        """Якщо pywin32 не встановлено — fallback на mspaint /pt."""
        with patch("core.printer._print_windows_win32", side_effect=ImportError("no pywin32")):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                _print_windows("test.jpg", "Printer")
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert args[0] == "mspaint"
                assert args[1] == "/pt"
                assert args[2] == "test.jpg"
                assert args[3] == "Printer"

    def test_win32print_error_falls_back_to_mspaint(self):
        """Якщо win32print кидає помилку — fallback на mspaint /pt."""
        with patch("core.printer._print_windows_win32", side_effect=RuntimeError("printer error")):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                _print_windows("test.jpg", "Printer")
                mock_run.assert_called_once()

    def test_mspaint_failure_raises_runtime_error(self):
        """Якщо mspaint /pt повертає ненульовий код — RuntimeError."""
        with patch("core.printer._print_windows_win32", side_effect=ImportError("no pywin32")):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="printer not found")
                with pytest.raises(RuntimeError) as exc_info:
                    _print_windows("test.jpg", "Printer")
                assert "printer not found" in str(exc_info.value)

    def test_no_printer_name_uses_shellexecute(self):
        """Якщо printer_name не задано і win32print недоступний — ShellExecute."""
        with patch("core.printer._print_windows_win32", side_effect=ImportError("no pywin32")):
            with patch("ctypes.windll.shell32.ShellExecuteW") as mock_shell:
                mock_shell.return_value = 42  # > 32 — успіх
                _print_windows("test.jpg", "")
                mock_shell.assert_called_once()

    def test_shellexecute_failure_raises(self):
        """Якщо ShellExecute повертає <= 32 — RuntimeError."""
        with patch("core.printer._print_windows_win32", side_effect=ImportError("no pywin32")):
            with patch("ctypes.windll.shell32.ShellExecuteW") as mock_shell:
                mock_shell.return_value = 0
                with pytest.raises(RuntimeError):
                    _print_windows("test.jpg", "")

    def test_mspaint_success_no_shellexecute(self):
        """Якщо mspaint /pt успішний — ShellExecute не викликається."""
        with patch("core.printer._print_windows_win32", side_effect=ImportError("no pywin32")):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                with patch("ctypes.windll.shell32.ShellExecuteW") as mock_shell:
                    _print_windows("test.jpg", "Printer")
                    mock_shell.assert_not_called()


# ============================================================
# Тест 4: _print_unix
# ============================================================

class TestPrintUnix:
    """Перевірка друку на Linux/Mac."""

    def test_lp_command_with_printer(self):
        """lp з -d printer_name."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _print_unix("test.jpg", "Printer")
            args = mock_run.call_args[0][0]
            assert args[0] == "lp"
            assert args[1] == "-d"
            assert args[2] == "Printer"
            assert args[3] == "test.jpg"

    def test_lp_command_without_printer(self):
        """lp без -d."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _print_unix("test.jpg", "")
            args = mock_run.call_args[0][0]
            assert args[0] == "lp"
            assert args[1] == "test.jpg"

    def test_lp_failure_raises(self):
        """lp повертає ненульовий код — RuntimeError."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="printer error")
            with pytest.raises(RuntimeError) as exc_info:
                _print_unix("test.jpg", "Printer")
            assert "printer error" in str(exc_info.value)


# ============================================================
# Тест 5: print_image — інтеграційний
# ============================================================

class TestPrintImage:
    """Перевірка print_image з моками."""

    def test_windows_calls_print_windows(self):
        """На Windows викликається _print_windows."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        with patch("sys.platform", "win32"):
            with patch("core.printer._print_windows") as mock_print:
                with patch("core.printer._delayed_remove") as mock_remove:
                    print_image(img, "Printer")
                    mock_print.assert_called_once()
                    mock_remove.assert_called_once()

    def test_unix_calls_print_unix(self):
        """На Linux/Mac викликається _print_unix."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        with patch("sys.platform", "linux"):
            with patch("core.printer._print_unix") as mock_print:
                with patch("os.remove") as mock_remove:
                    print_image(img, "Printer")
                    mock_print.assert_called_once()
                    mock_remove.assert_called_once()

    def test_error_propagates(self):
        """Помилка друку проброшується назовні."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        with patch("sys.platform", "win32"):
            with patch("core.printer._print_windows", side_effect=RuntimeError("print failed")):
                with patch("core.printer._delayed_remove"):
                    with pytest.raises(RuntimeError) as exc_info:
                        print_image(img, "Printer")
                    assert "print failed" in str(exc_info.value)