"""
Pipeline — єдина точка входу для GUI.
GUI викликає тільки pipeline, не знає про внутрішні модулі.
"""

import numpy as np
from processing import autofix, sharpen, hdr, perspective, brightness_contrast as bc


def run_autofix(
    image: np.ndarray,
    sharpen_strength: float = 0.4,
    hdr_strength: float = 0.5,
    use_hdr: bool = True,
    use_perspective: bool = True,
) -> np.ndarray:
    """
    Повний автоматичний pipeline:
    [Perspective avto] → LAB/CLAHE/Normalize → HDR → Sharpen
    """
    result = image

    if use_perspective:
        corrected, found = perspective.auto_correct(result)
        if found:
            result = corrected

    result = autofix.apply(
        result,
        sharpen_strength=sharpen_strength,
        hdr_strength=hdr_strength,
        use_hdr=use_hdr,
    )
    return result


def run_sharpen(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Тільки різкість."""
    return sharpen.apply(image, strength=strength)


def run_auto_sharpen(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Автоматична різкість: вимірює blur і застосовує, якщо потрібно.
    Повертає (результат, застосована_сила).
    """
    return sharpen.auto_apply(image)


def run_hdr(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Тільки HDR tone mapping."""
    return hdr.apply(image, strength=strength)


def run_perspective_auto(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Тільки авто-перспектива.
    Повертає (результат, знайдено).
    """
    return perspective.auto_correct(image)


def run_perspective_manual(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Перспектива за ручними точками.
    corners: float32 array shape (4,2).
    """
    return perspective.apply_correction(image, corners)


def detect_corners(image: np.ndarray) -> np.ndarray | None:
    """
    Повертає 4 кути документа (для відображення у GUI)
    або None якщо не знайдено.
    """
    return perspective.auto_detect_corners(image)


def run_brightness(image: np.ndarray, value: float) -> np.ndarray:
    """value: -1.0 … +1.0"""
    return bc.apply_brightness(image, value)


def run_auto_brightness(image: np.ndarray) -> np.ndarray:
    return bc.auto_brightness(image)


def run_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """value: -1.0 … +1.0"""
    return bc.apply_contrast(image, value)


def run_auto_contrast(image: np.ndarray) -> np.ndarray:
    return bc.auto_contrast(image)


def run_grayscale(image: np.ndarray) -> np.ndarray:
    return bc.to_grayscale(image)


def run_manual_adjustments(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    sharpen_strength: float = 0.0,
    hdr_strength: float = 0.0,
    grayscale: bool = False,
) -> np.ndarray:
    """Застосовує всі ручні корекції в правильному порядку."""
    result = image.copy()
    if grayscale:
        result = bc.to_grayscale(result)
    if abs(brightness) > 0.001:
        result = bc.apply_brightness(result, brightness)
    if abs(contrast) > 0.001:
        result = bc.apply_contrast(result, contrast)
    if hdr_strength > 0.001:
        result = hdr.apply(result, strength=hdr_strength)
    if sharpen_strength > 0.001:
        result = sharpen.apply(result, strength=sharpen_strength)
    return result
