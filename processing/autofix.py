"""
Auto Fix — автоматична корекція під фото документів.
Послідовність: LAB → CLAHE → Normalize → HDR tone mapping → Sharpen.
Не залежить від GUI модулів.
"""

import cv2
import numpy as np
from processing import hdr as hdr_module
from processing import sharpen as sharpen_module


def apply(
    image: np.ndarray,
    sharpen_strength: float = 0.4,
    hdr_strength: float = 0.5,
    use_hdr: bool = True,
) -> np.ndarray:
    """
    Повний Auto Fix pipeline.
    Повертає оброблений uint8 BGR.
    """
    result = _step_lab_clahe_normalize(image)
    if use_hdr:
        result = hdr_module.apply(result, strength=hdr_strength)
    result = sharpen_module.apply(result, strength=sharpen_strength)
    return result


def _step_lab_clahe_normalize(image: np.ndarray) -> np.ndarray:
    """
    1. Переводимо у LAB
    2. CLAHE на канал L (контраст)
    3. Normalize яскравості
    4. Повертаємо у BGR
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE — адаптивне вирівнювання гістограми
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_ch)

    # Normalize — розтягуємо яскравість на повний діапазон
    l_norm = cv2.normalize(l_clahe, None, 0, 255, cv2.NORM_MINMAX)

    merged = cv2.merge([l_norm, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
