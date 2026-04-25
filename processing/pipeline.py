"""
Pipeline — єдина точка входу для GUI.
GUI викликає тільки pipeline, не знає про внутрішні модулі.
"""

import numpy as np
from processing import autofix, sharpen, hdr, perspective, brightness_contrast as bc, doc_classifier


def run_autofix(
    image: np.ndarray,
    sharpen_strength: float = 0.4,
    hdr_strength: float = 0.5,
    use_hdr: bool = True,
    use_perspective: bool = True,
    doc_type: str | None = None,
    bw_binary: bool = False,
    classify_bw_std_thresh: float = 20.0,
    classify_edge_ratio_min: float = 0.03,
    classify_line_count_min: int = 3,
) -> tuple[np.ndarray, str]:
    """
    Повний автоматичний pipeline з авто-визначенням типу документа.
    Повертає (результат, статус_повідомлення).

    Типи:
      bw_document   — чб документ (без HDR, grayscale/bw, сильний контраст)
      color_document — кольоровий документ (без HDR, збереження кольорів)
      photo         — фото (повний pipeline з HDR)

    Параметр doc_type дозволяє примусово встановити тип;
    якщо None — тип визначається автоматично через doc_classifier.
    bw_binary — чи застосовувати адаптивну бінаризацію для bw_document.
    """
    result = image
    status_parts = []

    if use_perspective:
        corrected, found = perspective.auto_correct(result)
        if found:
            result = corrected
            status_parts.append("перспектива виправлена")

    if doc_type is None:
        doc_type = doc_classifier.classify(
            result,
            bw_std_thresh=classify_bw_std_thresh,
            edge_ratio_min=classify_edge_ratio_min,
            line_count_min=classify_line_count_min,
        )

    if doc_type == "bw_document":
        result = autofix.apply_bw_document(result, sharpen_strength=sharpen_strength, binary=bw_binary)
        status_parts.append("ч-б документ")
        if bw_binary:
            status_parts.append("бінаризація")
    elif doc_type == "color_document":
        result = autofix.apply_color_document(result, sharpen_strength=sharpen_strength)
        status_parts.append("кольоровий документ")
    else:
        # photo або fallback — повний pipeline
        result = autofix.apply(
            result,
            sharpen_strength=sharpen_strength,
            hdr_strength=hdr_strength,
            use_hdr=use_hdr,
        )
        status_parts.append("фото")
        if use_hdr:
            status_parts.append("HDR")

    status_parts.append(f"різкість {sharpen_strength:.2f}")
    status_msg = "Auto Fix: " + ", ".join(status_parts)
    return result, status_msg


def run_sharpen(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Тільки різкість."""
    return sharpen.apply(image, strength=strength)


def run_auto_sharpen(
    image: np.ndarray,
    threshold: float = 80.0,
    max_strength: float = 0.7,
) -> tuple[np.ndarray, float]:
    """
    Автоматична різкість: вимірює blur і застосовує, якщо потрібно.
    Повертає (результат, застосована_сила).
    """
    return sharpen.auto_apply(image, threshold=threshold, max_strength=max_strength)


def run_classify(
    image: np.ndarray,
    bw_std_thresh: float = 20.0,
    edge_ratio_min: float = 0.03,
    line_count_min: int = 3,
) -> str:
    """Повертає тип документа: 'bw_document' | 'color_document' | 'photo'."""
    return doc_classifier.classify(
        image,
        bw_std_thresh=bw_std_thresh,
        edge_ratio_min=edge_ratio_min,
        line_count_min=line_count_min,
    )


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


def run_auto_brightness(
    image: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> np.ndarray:
    return bc.auto_brightness(image, percentile_low=percentile_low, percentile_high=percentile_high)


def run_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """value: -1.0 … +1.0"""
    return bc.apply_contrast(image, value)


def run_auto_contrast(
    image: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> np.ndarray:
    return bc.auto_contrast(image, percentile_low=percentile_low, percentile_high=percentile_high)


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
