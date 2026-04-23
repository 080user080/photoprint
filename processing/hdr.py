"""
HDR tone mapping — локальне витягування деталей із тіней і світл.
Без злиття кількох знімків. Чисто numpy/OpenCV.
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np


def apply(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Простий HDR ефект через CLAHE у каналі яскравості (LAB).
    strength: 0.0 – без ефекту, 1.0 – максимальне витягування деталей.
    Повертає uint8 BGR.
    """
    if strength <= 0.0:
        return image.copy()

    strength = min(strength, 1.0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE з адаптивним clip_limit залежно від strength
    clip_limit = 1.0 + strength * 3.0   # 1.0 – 4.0
    tile_size = 8
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l_ch)

    # Blend між оригінальним L і обробленим залежно від strength
    l_result = cv2.addWeighted(l_ch, 1.0 - strength, l_eq, strength, 0)

    merged = cv2.merge([l_result, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
