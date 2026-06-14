"""
Завантаження зображень з конвертацією форматів → BGR numpy array.
Підтримує: JPG, PNG, WEBP, TIFF, HEIC/HEIF, RAW (CR2, NEF, ARW, DNG, та ін.).
Не залежить від GUI та processing модулів.
"""

import cv2
import numpy as np
import os
from utils.logger import get_logger


# Розширення RAW-форматів, що обробляються через rawpy
RAW_EXTENSIONS = {".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".srw",
                  ".raf", ".pef", ".x3f", ".3fr", ".raw"}

# Прапорець доступності rawpy — перевіряється при імпорті
try:
    import rawpy  # noqa: F401
    RAW_SUPPORTED = True
except ImportError:
    RAW_SUPPORTED = False


def _load_heic(path: str) -> np.ndarray:
    """Завантаження HEIC через pillow-heif → BGR numpy."""
    try:
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except ImportError as e:
        raise RuntimeError("pillow-heif не встановлено. pip install pillow-heif") from e


def _load_raw(path: str) -> np.ndarray:
    """
    Завантаження RAW-файлу через rawpy → RGB → BGR uint8.
    Повертає BGR numpy array uint8.
    Кидає RuntimeError, якщо rawpy не встановлено або файл не читається.
    """
    logger = get_logger(__name__)
    try:
        import rawpy
    except ImportError as e:
        raise RuntimeError(
            "rawpy не встановлено. pip install rawpy"
        ) from e

    try:
        with rawpy.imread(path) as raw:
            # postprocess з типовими налаштуваннями: auto white balance, gamma, що rawpy робить за замовчуванням
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        # rgb — uint8 HxWx3 RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        logger.debug(f"RAW завантажено: {path}, розмір: {bgr.shape}")
        return bgr
    except Exception as e:
        logger.error(f"Помилка читання RAW-файлу {path}: {e}")
        raise RuntimeError(f"Не вдалося прочитати RAW-файл: {path}") from e


def load(path: str) -> np.ndarray:
    """
    Завантажує зображення з диску.
    Повертає BGR numpy array uint8.
    Кидає RuntimeError якщо файл не вдалося прочитати.
    """
    logger = get_logger(__name__)
    if not os.path.isfile(path):
        logger.error(f"Файл не знайдено: {path}")
        raise RuntimeError(f"Файл не знайдено: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in (".heic", ".heif"):
        image = _load_heic(path)
    elif ext in RAW_EXTENSIONS:
        if not RAW_SUPPORTED:
            raise RuntimeError(
                "rawpy не встановлено. Неможливо завантажити RAW-файл.\n"
                "Встановіть rawpy: pip install rawpy"
            )
        image = _load_raw(path)
    else:
        # cv2.imread не підтримує unicode paths на Windows — використовуємо np.fromfile
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(f"Не вдалося декодувати зображення: {path}")
        raise RuntimeError(f"Не вдалося декодувати зображення: {path}")

    logger.debug(f"Завантажено зображення: {path}, розмір: {image.shape}")
    return image