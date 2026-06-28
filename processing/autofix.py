"""
Auto Fix — автоматична корекція під фото документів.
Послідовність: LAB → CLAHE → Normalize → HDR tone mapping → Sharpen.
Не залежить від GUI модулів.
"""

import cv2
import numpy as np
from processing import detail_map as detail_map_module
from processing import hdr as hdr_module
from processing import sharpen as sharpen_module
from processing import brightness_contrast as bc
from processing import text_mask as text_mask_module

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
    adaptive_hdr: bool = False,
) -> np.ndarray:
    """
    Повний Auto Fix pipeline для фотографій.
    adaptive_hdr: якщо True — використовує hdr.apply_adaptive з text_mask
                  для зменшення ефекту HDR у текстових областях.
    Повертає оброблений uint8 BGR.
    """
    result = _step_lab_clahe_normalize(image, aggressive=False)  # Не агресивний для фото
    if use_hdr:
        if adaptive_hdr:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            mask = text_mask_module.text_region_mask(gray)
            result = hdr_module.apply_adaptive(result, strength=hdr_strength, text_mask=mask)
        else:
            result = hdr_module.apply(result, strength=hdr_strength)
    result = sharpen_module.apply(result, strength=sharpen_strength)
    return result


def apply_bw_document(image: np.ndarray, sharpen_strength: float = 0.3, do_binary: bool = False,
                      skip_contrast: bool = False, skip_grayscale: bool = False) -> np.ndarray:
    """
    Pipeline для чорно-білих документів.
    Без HDR (щоб не псувати чіткість тексту).
    Послідовність: CLAHE → Auto-Contrast → Sharpen → Grayscale → [бінаризація].
    Параметр binary=False за замовчуванням (grayscale зберігає напівтони, бінаризація — чистий чорно-білий).

    Args:
        skip_contrast: якщо True — пропустити auto_contrast (для сумісності з pipeline, де contrast окремий крок)
        skip_grayscale: якщо True — пропустити to_grayscale (для сумісності з pipeline, де grayscale окремий крок)
    """
    result = _step_lab_clahe_normalize(image, aggressive=True)  # Агресивний для ч-б
    if not skip_contrast:
        result = bc.auto_contrast(result)
    result = sharpen_module.apply(result, strength=sharpen_strength)
    if not skip_grayscale:
        result = bc.to_grayscale(result)

    if do_binary:
        # Адаптивна бінаризація: збереже текст навіть при нерівному освітленні
        # adaptiveThreshold очікує grayscale (1 канал)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, BINARY_MAX_VALUE,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        BINARY_BLOCK_SIZE, BINARY_C)
        # Конвертуємо назад у BGR для сумісності з pipeline
        result = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
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
    # Используем resolution-independent tile grid
    tile_grid = hdr_module.adaptive_tile_grid(l_ch.shape[0], l_ch.shape[1])
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l_ch)

    if aggressive:
        # Агресивний normalize — тільки для ч-б документів
        l_norm = cv2.normalize(l_clahe, np.empty_like(l_clahe), 0, 255, cv2.NORM_MINMAX)
    else:
        # Фото/кольорові документи: CLAHE-ефект застосовуємо тільки там,
        # де є реальна локальна деталізація. На плоских/світлих ділянках
        # (detail_mask ≈ 0) результат лишається ≈ оригіналом.
        dmask = detail_map_module.detail_mask(l_ch)
        diff = hdr_module._apply_coring(l_clahe.astype(np.float32) - l_ch.astype(np.float32))
        l_norm = np.clip(l_ch.astype(np.float32) + dmask * diff, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_norm, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
