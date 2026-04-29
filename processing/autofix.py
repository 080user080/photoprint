"""
Auto Fix — автоматична корекція під фото документів.
Послідовність: LAB → CLAHE → Normalize → HDR tone mapping → Sharpen.
Не залежить від GUI модулів.
"""

import cv2
import numpy as np
from processing import hdr as hdr_module
from processing import sharpen as sharpen_module
from processing import brightness_contrast as bc

# Константи для CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# Константи для бінаризації
BINARY_MAX_VALUE = 255
BINARY_BLOCK_SIZE = 15
BINARY_C = 10


def apply(
    image: np.ndarray,
    sharpen_strength: float = 0.4,
    hdr_strength: float = 0.5,
    use_hdr: bool = True,
) -> np.ndarray:
    """
    Повний Auto Fix pipeline для фотографій.
    Повертає оброблений uint8 BGR.
    """
    result = _step_lab_clahe_normalize(image, aggressive=False)  # Не агресивний для фото
    if use_hdr:
        result = hdr_module.apply(result, strength=hdr_strength)
    result = sharpen_module.apply(result, strength=sharpen_strength)
    return result


def apply_bw_document(image: np.ndarray, sharpen_strength: float = 0.3, binary: bool = False) -> np.ndarray:
    """
    Pipeline для чорно-білих документів.
    Без HDR (щоб не псувати чіткість тексту).
    Послідовність: CLAHE → Auto-Contrast → Sharpen → Grayscale → [бінаризація].
    Параметр binary=False за замовчуванням (grayscale зберігає напівтони, бінаризація — чистий чорно-білий).
    """
    result = _step_lab_clahe_normalize(image, aggressive=True)  # Агресивний для ч-б
    result = bc.auto_contrast(result)
    result = sharpen_module.apply(result, strength=sharpen_strength)
    result = bc.to_grayscale(result)

    if binary:
        # Адаптивна бінаризація: збереже текст навіть при нерівному освітленні
        # adaptiveThreshold очікує grayscale (1 канал)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, BINARY_MAX_VALUE,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     BINARY_BLOCK_SIZE, BINARY_C)
        # Конвертуємо назад у BGR для сумісності з pipeline
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return result


def apply_color_document(image: np.ndarray, sharpen_strength: float = 0.2) -> np.ndarray:
    """
    Pipeline для кольорових документів (грамоти, посвідчення).
    Без HDR (щоб не змінювати кольори).
    Послідовність: CLAHE → Auto-Brightness → Auto-Contrast → легка Sharpen.
    """
    result = _step_lab_clahe_normalize(image, aggressive=False)  # Не агресивний для кольору
    result = bc.auto_brightness(result)
    result = bc.auto_contrast(result)
    result = sharpen_module.apply(result, strength=sharpen_strength)
    return result


def _step_lab_clahe_normalize(image: np.ndarray, aggressive: bool = True) -> np.ndarray:
    """
    1. Переводимо у LAB
    2. CLAHE на канал L (контраст)
    3. Normalize яскравості (якщо aggressive=True)
    4. Повертаємо у BGR

    Args:
        aggressive: Якщо True — робити глобальний normalize (для ч-б документів).
                   Якщо False — тільки CLAHE без глобального normalize (для фото/кольору).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE — адаптивне вирівнювання гістограми (безпечно для всіх)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    l_clahe = clahe.apply(l_ch)

    if aggressive:
        # Агресивний normalize — тільки для ч-б документів
        l_norm = cv2.normalize(l_clahe, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # Без глобального normalize — зберігаємо оригінальний діапазон
        l_norm = l_clahe

    merged = cv2.merge([l_norm, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
