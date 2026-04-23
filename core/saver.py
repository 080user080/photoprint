"""
Збереження зображення у JPG.
Не залежить від GUI та processing модулів.
"""

import cv2
import numpy as np
import os


def save(image: np.ndarray, path: str, quality: int = 95) -> str:
    """
    Зберігає BGR numpy array як JPG.
    Повертає шлях до збереженого файлу.
    Кидає RuntimeError якщо не вдалося зберегти.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    # cv2.imwrite не підтримує unicode paths на Windows
    success, buf = cv2.imencode(".jpg", image, params)
    if not success:
        raise RuntimeError(f"Не вдалося закодувати зображення у JPG: {path}")

    buf.tofile(path)
    return path
