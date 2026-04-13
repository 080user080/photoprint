"""
Відправка зображення на друк через priPrinter (або будь-який принтер системи).
Не залежить від GUI та processing модулів.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import cv2


def _save_temp_jpg(image: np.ndarray, quality: int = 95) -> str:
    """Зберігає зображення у тимчасовий JPG файл. Повертає шлях."""
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="photoprint_")
    os.close(fd)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", image, params)
    buf.tofile(path)
    return path


def print_image(image: np.ndarray, printer_name: str = "", jpg_quality: int = 95) -> None:
    """
    Відправляє зображення на друк.

    Windows: використовує ShellExecute з дієсловом 'print' або
             mspaint /pt для вибраного принтера.
    Linux/Mac: lp команда (для розробки/тестування).

    Кидає RuntimeError якщо не вдалося надіслати.
    """
    tmp_path = _save_temp_jpg(image, jpg_quality)

    try:
        if sys.platform == "win32":
            _print_windows(tmp_path, printer_name)
        else:
            _print_unix(tmp_path, printer_name)
    finally:
        # Видаляємо тимчасовий файл із затримкою (Windows блокує файл під час друку)
        if sys.platform != "win32":
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _print_windows(path: str, printer_name: str) -> None:
    """
    Друк на Windows.
    Якщо printer_name задано — використовує mspaint /pt (тихий друк на конкретний принтер).
    Інакше — ShellExecute 'print' (відкриває діалог з принтером за замовчуванням).
    """
    import ctypes

    if printer_name:
        # mspaint /pt <file> <printer> — тихий друк без діалогу
        result = subprocess.run(
            ["mspaint", "/pt", path, printer_name],
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mspaint /pt повернув код {result.returncode}. "
                f"Перевірте назву принтера: '{printer_name}'"
            )
    else:
        # ShellExecute 'print' — Windows сам обирає програму
        ret = ctypes.windll.shell32.ShellExecuteW(None, "print", path, None, None, 1)
        if ret <= 32:
            raise RuntimeError(f"ShellExecute 'print' повернув код {ret}")


def _print_unix(path: str, printer_name: str) -> None:
    """Друк на Linux/Mac через lp (для розробки)."""
    cmd = ["lp"]
    if printer_name:
        cmd += ["-d", printer_name]
    cmd.append(path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"lp помилка: {result.stderr}")
