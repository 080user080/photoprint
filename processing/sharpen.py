"""
Різкість зображення (Unsharp Mask).
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np


def apply(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """
    Unsharp Mask різкість.
    strength: 0.0 – без ефекту, 1.0 – максимум.
    Працює у float32, повертає uint8 BGR.
    """
    if strength <= 0.0:
        return image.copy()

    strength = min(strength, 1.0)

    # Gaussian blur як базова розмитість
    sigma = 1.0 + strength * 1.5        # 1.0 – 2.5
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Unsharp mask: оригінал + amount * (оригінал - розмите)
    amount = 0.5 + strength * 1.0       # 0.5 – 1.5
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    return sharpened
