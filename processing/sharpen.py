"""
Різкість зображення (Unsharp Mask).
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для Unsharp Mask
SIGMA_BASE = 1.0
SIGMA_MULTIPLIER = 1.5  # sigma = 1.0 + strength * 1.5
AMOUNT_BASE = 0.5
AMOUNT_MULTIPLIER = 1.0  # amount = 0.5 + strength * 1.0
MAX_STRENGTH = 1.0

# Константи для вимірювання різкості
MEASURE_MAX_SIDE = 500  # максимальний розмір для стабільності виміру

# Константи для авто-різкості
MIN_AUTO_STRENGTH = 0.15  # мінімальна сила, щоб був помітний ефект


def apply(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """
    Unsharp Mask різкість.
    strength: 0.0 – без ефекту, 1.0 – максимум.
    Працює у float32, повертає uint8 BGR.
    """
    if strength <= 0.0:
        return image.copy()

    strength = min(strength, MAX_STRENGTH)

    # Gaussian blur як базова розмитість
    sigma = SIGMA_BASE + strength * SIGMA_MULTIPLIER
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Unsharp mask: оригінал + amount * (оригінал - розмите)
    amount = AMOUNT_BASE + strength * AMOUNT_MULTIPLIER
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    return sharpened


def measure_sharpness(image: np.ndarray) -> float:
    """
    Вимірює різкість зображення через variance of Laplacian.
    Вище значення = різкіше зображення.
    Працює на каналі L (LAB), щоб не реагувати на кольорові шуми.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    # Зменшуємо до MEASURE_MAX_SIDE для стабільності виміру незалежно від розміру
    h, w = l.shape[:2]
    if max(h, w) > MEASURE_MAX_SIDE:
        scale = MEASURE_MAX_SIDE / max(h, w)
        l = cv2.resize(l, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(l, cv2.CV_64F)
    return float(np.var(lap))


def auto_apply(image: np.ndarray, threshold: float = 80.0, max_strength: float = 0.7) -> tuple[np.ndarray, float]:
    """
    Автоматична різкість: вимірює blur і застосовує різкість, якщо зображення розмите.
    Повертає (результат, застосована_сила).
    """
    variance = measure_sharpness(image)
    if variance >= threshold:
        # Вже достатньо різке — нічого не робимо
        return image.copy(), 0.0

    # Чим менше variance, тим більша різкість (лінійна інтерполяція)
    # variance=0   → max_strength
    # variance=threshold → 0.0
    strength = max_strength * (1.0 - variance / threshold)
    strength = max(MIN_AUTO_STRENGTH, min(strength, max_strength))

    return apply(image, strength=strength), strength
