"""
Завантаження зображень з конвертацією форматів → BGR numpy array.
Підтримує: JPG, PNG, WEBP, TIFF, HEIC/HEIF.
Не залежить від GUI та processing модулів.
"""

import cv2
import numpy as np
import os
from utils.logger import get_logger


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
    else:
        # cv2.imread не підтримує unicode paths на Windows — використовуємо np.fromfile
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(f"Не вдалося декодувати зображення: {path}")
        raise RuntimeError(f"Не вдалося декодувати зображення: {path}")

    logger.debug(f"Завантажено зображення: {path}, розмір: {image.shape}")
    return image
