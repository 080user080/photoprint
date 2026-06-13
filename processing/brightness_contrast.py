"""
Яскравість і контраст.
Всі операції — numpy/OpenCV, без Python-циклів.
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для яскравості
BRIGHTNESS_THRESHOLD = 0.001
BRIGHTNESS_SHIFT_MULTIPLIER = 100.0  # shift = value * 100
AUTO_BRIGHTNESS_MIN_RANGE = 10  # мінімальний діапазон для авто-корекції

# Константи для контрасту
CONTRAST_THRESHOLD = 0.001
CONTRAST_ALPHA_POS_MULTIPLIER = 2.0  # alpha = 1.0 + value * 2.0 (value >= 0)
CONTRAST_ALPHA_NEG_MULTIPLIER = 0.8  # alpha = 1.0 + value * 0.8 (value < 0)
CONTRAST_CENTER = 128.0

# Константи для LUT
LUT_SIZE = 256
LUT_MIN = 0
LUT_MAX = 255


# ---------------------------------------------------------------------------
# Яскравість
# ---------------------------------------------------------------------------

def apply_brightness(image: np.ndarray, value: float) -> np.ndarray:
    """
    value: -1.0 (темно) … 0.0 (без змін) … +1.0 (світло).
    Конвертує у float32, додає зміщення, клампує у 0-255.
    """
    if abs(value) < BRIGHTNESS_THRESHOLD:
        return image.copy()
    shift = value * BRIGHTNESS_SHIFT_MULTIPLIER
    lut = _make_brightness_lut(shift)
    return cv2.LUT(image, lut)


def auto_brightness(image: np.ndarray, percentile_low: float = 5.0, percentile_high: float = 95.0) -> np.ndarray:
    """
    Автояскравість: розтягує percentile_low–percentile_high перцентилі каналу L на повний діапазон 0–255.
    Працює у LAB. Ігнорує крайні викиди.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    p_low  = float(np.percentile(l, percentile_low))
    p_high = float(np.percentile(l, percentile_high))

    # Якщо діапазон вже достатній — нічого не робимо
    if p_high - p_low < AUTO_BRIGHTNESS_MIN_RANGE:
        return image.copy()

    # Розтягуємо на 0–255
    l_stretch = np.clip((l.astype(np.float32) - p_low) / (p_high - p_low) * LUT_MAX, LUT_MIN, LUT_MAX).astype(np.uint8)

    merged = cv2.merge([l_stretch, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Контраст
# ---------------------------------------------------------------------------

def apply_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """
    value: -1.0 (низький) … 0.0 (без змін) … +1.0 (високий).
    Використовує лінійне масштабування навколо 128.
    """
    if abs(value) < CONTRAST_THRESHOLD:
        return image.copy()
    # alpha: 0.2 (мінімум) … 1.0 (нейтраль) … 3.0 (максимум)
    if value >= 0:
        alpha = 1.0 + value * CONTRAST_ALPHA_POS_MULTIPLIER
    else:
        alpha = 1.0 + value * CONTRAST_ALPHA_NEG_MULTIPLIER
    lut = _make_contrast_lut(alpha)
    return cv2.LUT(image, lut)


def auto_contrast(image: np.ndarray, percentile_low: float = 5.0, percentile_high: float = 95.0) -> np.ndarray:
    """
    Авто-контраст: розтягує гістограму каналу L на percentile_low–percentile_high перцентилі.
    Ігнорує крайні викиди — не пересвітлює.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p_low  = float(np.percentile(l, percentile_low))
    p_high = float(np.percentile(l, percentile_high))
    if p_high - p_low < AUTO_BRIGHTNESS_MIN_RANGE:
        return image.copy()
    l_norm = np.clip((l.astype(np.float32) - p_low) / (p_high - p_low) * LUT_MAX, LUT_MIN, LUT_MAX).astype(np.uint8)
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
    lut = np.arange(LUT_SIZE, dtype=np.float32) + shift
    lut = np.clip(lut, LUT_MIN, LUT_MAX).astype(np.uint8)
    return lut


def _make_contrast_lut(alpha: float) -> np.ndarray:
    lut = (np.arange(LUT_SIZE, dtype=np.float32) - CONTRAST_CENTER) * alpha + CONTRAST_CENTER
    lut = np.clip(lut, LUT_MIN, LUT_MAX).astype(np.uint8)
    return lut
