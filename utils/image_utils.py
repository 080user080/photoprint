"""
Утиліти для роботи із зображеннями (numpy/OpenCV).
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для прев'ю
DEFAULT_PREVIEW_MAX_SIDE = 900


def make_preview(image: np.ndarray, max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> np.ndarray:
    """
    Повертає зменшену копію для прев'ю.
    Оригінальне зображення не змінюється.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image.copy()
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV → RGB для відображення у Qt."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Гарантує 3-канальний BGR (конвертує з grayscale якщо треба)."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
