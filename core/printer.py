"""
Відправка зображення на друк через priPrinter (або будь-який принтер системи).
Не залежить від GUI та processing модулів.
"""

import os
import sys
import tempfile
import subprocess
import threading
import numpy as np
import cv2
from utils.logger import get_logger

# Константи
DEFAULT_JPG_QUALITY = 95
TEMP_FILE_PREFIX = "photoprint_"
TEMP_FILE_SUFFIX = ".jpg"
SHELLEXECUTE_SUCCESS_MIN = 32
TEMP_FILE_DELETE_DELAY_SEC = 5.0


def _save_temp_jpg(image: np.ndarray, quality: int = DEFAULT_JPG_QUALITY) -> str:
    """Зберігає зображення у тимчасовий JPG файл. Повертає шлях."""
    fd, path = tempfile.mkstemp(suffix=TEMP_FILE_SUFFIX, prefix=TEMP_FILE_PREFIX)
    os.close(fd)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(TEMP_FILE_SUFFIX, image, params)
    buf.tofile(path)
    return path


def _delayed_remove(path: str, delay: float = TEMP_FILE_DELETE_DELAY_SEC) -> None:
    """Відкладене видалення файлу через delay секунд (для Windows, де файл блокується)."""
    def _remove():
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    threading.Timer(delay, _remove).start()


def print_image(image: np.ndarray, printer_name: str = "", jpg_quality: int = DEFAULT_JPG_QUALITY) -> None:
    """
    Відправляє зображення на друк.

    Windows: використовує ShellExecute з дієсловом 'print' або
             mspaint /pt для вибраного принтера.
    Linux/Mac: lp команда (для розробки/тестування).

    Кидає RuntimeError якщо не вдалося надіслати.
    """
    logger = get_logger(__name__)
    tmp_path = _save_temp_jpg(image, jpg_quality)
    logger.info(f"Відправка на друк: {tmp_path}, принтер: {printer_name or 'за замовчуванням'}")

    try:
        if sys.platform == "win32":
            _print_windows(tmp_path, printer_name)
        else:
            _print_unix(tmp_path, printer_name)
        logger.info("Зображення успішно надіслано на друк")
    finally:
        # Видаляємо тимчасовий файл із затримкою (Windows блокує файл під час друку)
        if sys.platform == "win32":
            _delayed_remove(tmp_path)
        else:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _print_windows_win32(path: str, printer_name: str) -> None:
    """
    Друк через win32print/win32ui (надійний, без mspaint).
    Використовує pywin32 для прямого друку зображення на принтер.

    Кидає ImportError якщо pywin32 не встановлено.
    Кидає RuntimeError якщо друк не вдався.
    """
    import win32print
    import win32ui
    from typing import Any
    from PIL import Image, ImageWin

    # Якщо printer_name не задано — використовуємо принтер за замовчуванням
    if not printer_name:
        printer_name = win32print.GetDefaultPrinter()

    hprinter = win32print.OpenPrinter(printer_name)
    try:
        # win32ui.CreateDC() не має типових стабів у Pylance (повертає None),
        # тому анотуємо як Any, щоб уникнути хибних reportAttributeAccessIssue.
        hdc: Any = win32ui.CreateDC()
        hdc.CreatePrinterDC(printer_name)
        hdc.StartDoc(path)
        hdc.StartPage()

        img = Image.open(path)
        dib = ImageWin.Dib(img)

        # Отримуємо розміри друкованої області (HORZRES=110, VERTRES=111)
        printable_w = hdc.GetDeviceCaps(110)
        printable_h = hdc.GetDeviceCaps(111)

        # Масштабуємо зображення до розміру друкованої області
        scale = min(printable_w / img.width, printable_h / img.height)
        draw_w = int(img.width * scale)
        draw_h = int(img.height * scale)

        # Малюємо зображення в пам'яті
        mem_dc: Any = hdc.CreateCompatibleDC()
        bmp: Any = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(hdc, img.width, img.height)
        mem_dc.SelectObject(bmp)
        dib.draw(mem_dc.GetHandleOutput(), (0, 0, img.width, img.height))

        # Копіюємо на принтер з масштабуванням (SRCCOPY = 0x00CC0020)
        hdc.StretchBlt((0, 0, draw_w, draw_h), mem_dc, (0, 0, img.width, img.height), 0x00CC0020)

        hdc.EndPage()
        hdc.EndDoc()
        hdc.DeleteDC()
    finally:
        win32print.ClosePrinter(hprinter)


def _print_windows(path: str, printer_name: str) -> None:
    """
    Друк на Windows.
    Ланцюжок fallback:
      1. win32print (якщо pywin32 доступний) — надійний друк
      2. mspaint /pt (якщо printer_name задано)
      3. ShellExecute 'print' (якщо printer_name не задано)
    """
    logger = get_logger(__name__)
    import ctypes

    # 1. Спробувати win32print (якщо pywin32 доступний)
    try:
        _print_windows_win32(path, printer_name)
        logger.info("Друк через win32print успішний")
        return
    except ImportError:
        logger.info("pywin32 не встановлено, використовуємо fallback")
    except Exception as e:
        logger.warning(f"win32print не вдалося: {e}, використовуємо fallback")

    # 2. mspaint /pt (якщо printer_name задано)
    if printer_name:
        result = subprocess.run(
            ["mspaint", "/pt", path, printer_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("Друк через mspaint /pt успішний")
            return
        stderr = result.stderr.strip() if result.stderr else ""
        logger.error(f"mspaint /pt повернув код {result.returncode}, принтер: {printer_name}")
        msg = (
            f"Помилка друку: принтер '{printer_name}' недоступний або не знайдений.\n"
            f"Перевірте назву принтера в Налаштуваннях.\n"
            f"Технічна інформація: mspaint /pt повернув код {result.returncode}."
        )
        if stderr:
            msg += f"\n{stderr}"
        raise RuntimeError(msg)

    # 3. ShellExecute 'print' — Windows сам обирає програму
    ret = ctypes.windll.shell32.ShellExecuteW(None, "print", path, None, None, 1)
    if ret <= SHELLEXECUTE_SUCCESS_MIN:
        logger.error(f"ShellExecute 'print' повернув код {ret}")
        raise RuntimeError(f"ShellExecute 'print' повернув код {ret}")
    logger.info("Друк через ShellExecute успішний")


def _print_unix(path: str, printer_name: str) -> None:
    """Друк на Linux/Mac через lp (для розробки)."""
    logger = get_logger(__name__)
    cmd = ["lp"]
    if printer_name:
        cmd += ["-d", printer_name]
    cmd.append(path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"lp помилка: {result.stderr}")
        raise RuntimeError(f"lp помилка: {result.stderr}")
