"""
Яскравість і контраст.
Всі операції — numpy/OpenCV, без Python-циклів.
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Яскравість
# ---------------------------------------------------------------------------

def apply_brightness(image: np.ndarray, value: float) -> np.ndarray:
    """
    value: -1.0 (темно) … 0.0 (без змін) … +1.0 (світло).
    Конвертує у float32, додає зміщення, клампує у 0-255.
    """
    if abs(value) < 0.001:
        return image.copy()
    shift = value * 100.0          # -100 … +100
    lut = _make_brightness_lut(shift)
    return cv2.LUT(image, lut)


def auto_brightness(image: np.ndarray) -> np.ndarray:
    """
    Автояскравість: зміщує середню яскравість каналу L до 128.
    Працює у LAB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = float(np.mean(l))
    shift = 128.0 - mean_l
    lut = _make_brightness_lut(shift)
    l_fixed = cv2.LUT(l, lut)
    merged = cv2.merge([l_fixed, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Контраст
# ---------------------------------------------------------------------------

def apply_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """
    value: -1.0 (низький) … 0.0 (без змін) … +1.0 (високий).
    Використовує лінійне масштабування навколо 128.
    """
    if abs(value) < 0.001:
        return image.copy()
    # alpha: 0.2 (мінімум) … 1.0 (нейтраль) … 3.0 (максимум)
    if value >= 0:
        alpha = 1.0 + value * 2.0
    else:
        alpha = 1.0 + value * 0.8   # 0.2 при value=-1
    lut = _make_contrast_lut(alpha)
    return cv2.LUT(image, lut)


def auto_contrast(image: np.ndarray) -> np.ndarray:
    """
    Авто-контраст: розтягує гістограму каналу L на 5–95 перцентилі
    (ігнорує крайні значення щоб не пересвітлювати).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p_low  = float(np.percentile(l, 2))
    p_high = float(np.percentile(l, 98))
    if p_high - p_low < 10:
        return image.copy()
    l_norm = np.clip((l.astype(np.float32) - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    merged = cv2.merge([l_norm, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Чорно-білий режим
# ---------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Конвертує у grayscale і повертає 3-канальний BGR
    (щоб подальший pipeline не ламався).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# LUT-хелпери (швидше ніж поелементні операції)
# ---------------------------------------------------------------------------

def _make_brightness_lut(shift: float) -> np.ndarray:
    lut = np.arange(256, dtype=np.float32) + shift
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut


def _make_contrast_lut(alpha: float) -> np.ndarray:
    lut = (np.arange(256, dtype=np.float32) - 128.0) * alpha + 128.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut
