"""
Pipeline — єдина точка входу для GUI.
GUI викликає тільки pipeline, не знає про внутрішні модулі.
"""

from enum import Enum
from typing import Optional
import cv2
import numpy as np
from processing import autofix, sharpen, hdr, perspective, brightness_contrast as bc, doc_classifier, shadow_highlight, shadow_remove


class DocType(str, Enum):
    """Типи документів для класифікації."""
    BW_DOCUMENT = "bw_document"
    COLOR_DOCUMENT = "color_document"
    PHOTO = "photo"


# Константи для порогів застосування корекцій
EPSILON = 0.001  # Поріг для ігнорування дуже малих значень


def run_contrast_advanced(image: np.ndarray, value: float, mode: str = "linear") -> np.ndarray:
    """
    Застосовує контраст вибраним методом.
    mode: "linear", "percentile", "s_curve", "adaptive".
    Для нелінійних методів використовується тільки додатна сила (0..1).
    """
    if abs(value) < EPSILON:
        return image.copy()
    if mode == "linear":
        return bc.apply_contrast(image, value)
    # Для нелінійних методів — тільки додатна сила
    strength = max(0.0, value)
    if strength < EPSILON:
        return image.copy()
    if mode == "percentile":
        return bc.smart_contrast_percentile(image, strength=strength)
    elif mode == "s_curve":
        return bc.contrast_s_curve(image, strength=strength)
    elif mode == "adaptive":
        return bc.local_contrast_adaptive(image, strength=strength)
    else:
        return bc.apply_contrast(image, value)


def run_autofix(
    image: np.ndarray,
    sharpen_strength: float = 0.4,
    hdr_strength: float = 0.5,
    use_hdr: bool = True,
    use_perspective: bool = True,
    partial_perspective: bool = False,
    doc_type: Optional[str] = None,
    bw_binary: bool = False,
    classify_bw_std_thresh: float = 20.0,
    classify_edge_ratio_min: float = 0.03,
    classify_line_count_min: int = 3,
    shadow_highlight_strength: float = 0.0,
    output_color_mode: str = "auto",
    adaptive_hdr: bool = False,
    autofix_contrast: float = 0.15,
    contrast_mode: str = "linear",
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
    shadow_highlight_strength — сила висвітлення тіней (0-2.0).
    output_color_mode — формат виходу: "auto" (за типом), "color", "grayscale", "binary".
    adaptive_hdr — якщо True, використовує hdr.apply_adaptive для фото-документів.
    contrast_mode — метод контрасту ("linear", "percentile", "s_curve", "adaptive").
    """
    result = image.copy()
    status_parts = []

    # Спочатку визначаємо тип документа (до будь-якої обробки!)
    if doc_type is None:
        doc_type = doc_classifier.classify(
            result,
            bw_std_thresh=classify_bw_std_thresh,
            edge_ratio_min=classify_edge_ratio_min,
            line_count_min=classify_line_count_min,
        )

    # Видалення тіней — ТІЛЬКИ для ч-б документів (руйнує кольорові/фото!)
    if doc_type == DocType.BW_DOCUMENT.value:
        result, had_shadow = shadow_remove.auto_remove_shadow(result)
        if had_shadow:
            status_parts.append("тіні видалено")

    # Висвітлення тіней — додаткове підсвічування
    if shadow_highlight_strength > EPSILON:
        result = shadow_highlight.apply_shadow_highlight(result, strength=shadow_highlight_strength)
        status_parts.append(f"підсвічування {shadow_highlight_strength:.2f}")

    if use_perspective:
        corrected, found = perspective.auto_correct(result) if not partial_perspective else perspective.auto_correct_partial(result)
        if found:
            result = corrected
            status_parts.append("перспектива виправлена")

    if doc_type == DocType.BW_DOCUMENT.value:
        result = autofix.apply_bw_document(result, sharpen_strength=sharpen_strength, binary=bw_binary)
        status_parts.append("ч-б документ")
        if bw_binary:
            status_parts.append("бінаризація")
    elif doc_type == DocType.COLOR_DOCUMENT.value:
        result = autofix.apply_color_document(result, sharpen_strength=sharpen_strength)
        status_parts.append("кольоровий документ")
    else:
        # photo або fallback — повний pipeline
        result = autofix.apply(
            result,
            sharpen_strength=sharpen_strength,
            hdr_strength=hdr_strength,
            use_hdr=use_hdr,
            adaptive_hdr=adaptive_hdr,
        )
        status_parts.append("фото")
        if use_hdr:
            if adaptive_hdr:
                status_parts.append("адаптивний HDR")
            else:
                status_parts.append("HDR")

    status_parts.append(f"різкість {sharpen_strength:.2f}")

    # Додатковий контраст в кінці циклу (налаштовується)
    result = run_contrast_advanced(result, autofix_contrast, contrast_mode)

    # Формат виходу: якщо не "auto" — примусово конвертуємо
    if output_color_mode == "grayscale":
        result = bc.to_grayscale(result)
        status_parts.append("ч-б")
    elif output_color_mode == "binary":
        result = bc.to_grayscale(result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
        result = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        status_parts.append("бінаризація")
    elif output_color_mode == "color":
        # Нічого не робимо — залишаємо кольоровим
        pass
    # "auto" — залишаємо як є (визначено типом документа)

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
    """Повертає тип документа: DocType.BW_DOCUMENT | DocType.COLOR_DOCUMENT | DocType.PHOTO."""
    return doc_classifier.classify(
        image,
        bw_std_thresh=bw_std_thresh,
        edge_ratio_min=edge_ratio_min,
        line_count_min=line_count_min,
    )


def run_hdr(image: np.ndarray, strength: float = 0.5, adaptive: bool = False) -> np.ndarray:
    """Тільки HDR tone mapping.
    adaptive: якщо True — використовує hdr.apply_adaptive з text_mask.
    """
    if adaptive:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        from processing import text_mask as text_mask_module
        mask = text_mask_module.text_region_mask(gray)
        return hdr.apply_adaptive(image, strength=strength, text_mask=mask)
    return hdr.apply(image, strength=strength)


def run_perspective_auto(image: np.ndarray, partial: bool = False) -> tuple[np.ndarray, bool]:
    """
    Тільки авто-перспектива.
    Повертає (результат, знайдено).
    partial: якщо True — використовує apply_partial_correction.
    """
    if partial:
        return perspective.auto_correct_partial(image)
    return perspective.auto_correct(image)


def run_perspective_manual(image: np.ndarray, corners: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Перспектива за ручними точками.
    corners: float32 array shape (4,2).
    partial: якщо True — використовує apply_partial_correction.
    """
    if partial:
        return perspective.apply_partial_correction(image, corners)
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


def run_shadow_remove(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Видалення градієнтних тіней з документа.
    Повертає (результат, чи_були_тіні).
    """
    return shadow_remove.auto_remove_shadow(image)


def run_full_auto(
    image: np.ndarray,
    settings: dict,
    dry_run: bool = False,
) -> tuple[np.ndarray, str, dict]:
    """
    Адаптивний Full Auto pipeline.
    Аналізує зображення через diagnostics.diagnose і застосовує
    тільки потрібні корекції з адаптивною силою.

    Параметр dry_run=True — тільки діагностика, нічого не застосовувати.
    Повертає (result, status_message, applied_steps).
    """
    from processing import diagnostics

    result = image.copy()
    applied_steps: dict[str, float | bool] = {}
    status_parts: list[str] = []

    # Крок 0 — Діагностика
    diag = diagnostics.diagnose(image, settings)

    if dry_run:
        status_parts.append("тільки діагностика")
        status_parts.append(f"тип: {diag.doc_type}")
        if diag.gradient_has:
            status_parts.append(f"градієнт: {diag.gradient_direction} ({diag.gradient_strength:.2f})")
        status_parts.append(f"контраст: {diag.contrast_strength_needed:.2f}")
        status_parts.append(f"яскравість: {diag.brightness_correction:.2f}")
        status_parts.append(f"розмиття: {diag.blur_strength_needed:.2f}")
        if diag.perspective_has:
            status_parts.append(f"перспектива: {diag.perspective_skew_ratio:.3f}")
        status_msg = "Full Auto (dry): " + ", ".join(status_parts)
        return image.copy(), status_msg, applied_steps

    # Крок 1 — Видалення градієнтного фону
    min_gradient_strength = settings.get("full_auto_min_gradient_strength", 0.3)
    if diag.gradient_has and diag.doc_type != DocType.PHOTO.value and diag.gradient_strength > min_gradient_strength:
        result, had_shadow = shadow_remove.auto_remove_shadow(result)
        if had_shadow:
            applied_steps["shadow_remove"] = diag.gradient_strength
            status_parts.append("тіні видалено")
            # Частковий перерахунок
            updated = diagnostics.partial_rediagnose(result, settings, ["contrast", "brightness"])
            diag.contrast_range_l = updated.get("contrast_range_l", diag.contrast_range_l)
            diag.contrast_strength_needed = updated.get("contrast_strength_needed", diag.contrast_strength_needed)
            diag.brightness_mean_l = updated.get("brightness_mean_l", diag.brightness_mean_l)
            diag.brightness_correction = updated.get("brightness_correction", diag.brightness_correction)

    # Крок 2 — Корекція перспективи
    if diag.perspective_has:
        result = perspective.apply_correction(result, diag.perspective_corners)
        applied_steps["perspective"] = diag.perspective_skew_ratio
        status_parts.append("перспектива")
        # Частковий перерахунок
        updated = diagnostics.partial_rediagnose(result, settings, ["contrast", "brightness", "blur"])
        diag.contrast_range_l = updated.get("contrast_range_l", diag.contrast_range_l)
        diag.contrast_strength_needed = updated.get("contrast_strength_needed", diag.contrast_strength_needed)
        diag.brightness_mean_l = updated.get("brightness_mean_l", diag.brightness_mean_l)
        diag.brightness_correction = updated.get("brightness_correction", diag.brightness_correction)
        diag.blur_variance = updated.get("blur_variance", diag.blur_variance)
        diag.blur_strength_needed = updated.get("blur_strength_needed", diag.blur_strength_needed)
        diag.blur_sharpen_strength = updated.get("blur_sharpen_strength", diag.blur_sharpen_strength)

    # Крок 3 — Яскравість
    brightness_strength = abs(diag.brightness_correction)
    if brightness_strength >= 0.05:
        if diag.brightness_correction > 0:
            # Темне — агресивніше розтягування
            result = bc.auto_brightness(result, percentile_low=2.0, percentile_high=98.0)
        else:
            # Світле — м'якше
            result = bc.auto_brightness(result, percentile_low=5.0, percentile_high=95.0)
        applied_steps["brightness"] = brightness_strength
        status_parts.append(f"яскравість {brightness_strength:.2f}")
        # Частковий перерахунок контрасту
        updated = diagnostics.partial_rediagnose(result, settings, ["contrast"])
        diag.contrast_strength_needed = updated.get("contrast_strength_needed", diag.contrast_strength_needed)

    # Крок 4 — Контраст
    contrast_strength = diag.contrast_strength_needed
    if contrast_strength >= 0.05:
        contrast_strength = min(contrast_strength, 0.85)
        contrast_mode = settings.get("contrast_mode", "linear")
        result = run_contrast_advanced(result, contrast_strength, contrast_mode)
        applied_steps["contrast"] = contrast_strength
        status_parts.append(f"контраст {contrast_strength:.2f}")
        # Частковий перерахунок розмиття
        updated = diagnostics.partial_rediagnose(result, settings, ["blur"])
        diag.blur_sharpen_strength = updated.get("blur_sharpen_strength", diag.blur_sharpen_strength)

    # Крок 5 — HDR (тільки для фото)
    if diag.doc_type == DocType.PHOTO.value and settings.get("hdr_in_autofix", True):
        hdr_strength = settings.get("hdr_strength", 0.5)
        result = hdr.apply_adaptive(result, strength=hdr_strength)
        applied_steps["hdr"] = hdr_strength
        status_parts.append(f"HDR {hdr_strength:.2f}")

    # Крок 6 — Специфічна обробка по типу документа
    sharpen_strength = diag.blur_sharpen_strength
    if sharpen_strength <= 0.0:
        sharpen_strength = settings.get("sharpen_strength", 0.4)
    bw_binary = settings.get("bw_binary", False)

    if diag.doc_type == DocType.BW_DOCUMENT.value:
        result = autofix.apply_bw_document(result, sharpen_strength=sharpen_strength, binary=bw_binary)
        status_parts.append("ч-б документ")
    elif diag.doc_type == DocType.COLOR_DOCUMENT.value:
        result = autofix.apply_color_document(result, sharpen_strength=sharpen_strength)
        status_parts.append("кольоровий документ")
    else:
        if diag.blur_strength_needed > 0.05:
            result = sharpen.apply(result, strength=sharpen_strength)
            status_parts.append(f"різкість {sharpen_strength:.2f}")
        status_parts.append("фото")
    applied_steps["doc_processing"] = diag.doc_type

    # Крок 7 — Shadow highlight
    sh_strength = settings.get("shadow_highlight_strength", 0.0)
    if sh_strength > 0.001:
        result = shadow_highlight.apply_shadow_highlight(result, strength=sh_strength)
        applied_steps["shadow_highlight"] = sh_strength
        status_parts.append(f"підсвічування {sh_strength:.2f}")

    # Крок 8 — Додатковий контраст Auto Fix
    autofix_contrast = settings.get("autofix_contrast", 0.15)
    contrast_mode = settings.get("contrast_mode", "linear")
    if autofix_contrast > 0.001:
        result = run_contrast_advanced(result, autofix_contrast, contrast_mode)
        applied_steps["autofix_contrast"] = autofix_contrast

    # Крок 9 — Формат виходу
    output_color_mode = settings.get("output_color_mode", "auto")
    if output_color_mode == "grayscale":
        result = bc.to_grayscale(result)
        status_parts.append("ч-б")
    elif output_color_mode == "binary":
        result = bc.to_grayscale(result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
        result = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        status_parts.append("бінаризація")

    status_msg = "Full Auto: " + ", ".join(status_parts)
    return result, status_msg, applied_steps


def run_manual_adjustments(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    sharpen_strength: float = 0.0,
    hdr_strength: float = 0.0,
    grayscale: bool = False,
    shadow_highlight_strength: float = 0.0,
    adaptive_hdr: bool = False,
    contrast_mode: str = "linear",
) -> np.ndarray:
    """Застосовує всі ручні корекції в правильному порядку.

    УВАГА: shadow_remove НЕ викликається тут — він викликається тільки
    в run_autofix для BW документів. Повторний виклик при зміні слайдера
    призводив до "чорної інверсії" через overflow у cv2.divide.
    """
    result = image.copy()
    # Висвітлення тіней — додаткове підсвічування
    if shadow_highlight_strength > EPSILON:
        result = shadow_highlight.apply_shadow_highlight(result, strength=shadow_highlight_strength)
    if grayscale:
        result = bc.to_grayscale(result)
    if abs(brightness) > EPSILON:
        result = bc.apply_brightness(result, brightness)
    if abs(contrast) > EPSILON:
        result = run_contrast_advanced(result, contrast, contrast_mode)
    if hdr_strength > EPSILON:
        if adaptive_hdr:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            from processing import text_mask as text_mask_module
            mask = text_mask_module.text_region_mask(gray)
            result = hdr.apply_adaptive(result, strength=hdr_strength, text_mask=mask)
        else:
            result = hdr.apply(result, strength=hdr_strength)
    if sharpen_strength > EPSILON:
        result = sharpen.apply(result, strength=sharpen_strength)
    return result