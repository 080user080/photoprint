"""
Pipeline — єдина точка входу для GUI.
GUI викликає тільки pipeline, не знає про внутрішні модулі.
"""

import logging
from enum import Enum
from typing import Optional
import cv2
import numpy as np
from processing import autofix, sharpen, hdr, perspective, brightness_contrast as bc, doc_classifier, shadow_highlight, shadow_remove, white_background, deskew as deskew_module, color_cast
from processing.doc_classifier import CAPTURE_PHONE, CAPTURE_SCREEN

logger = logging.getLogger(__name__)


class DocType(str, Enum):
    """Типи документів для класифікації."""
    BW_DOCUMENT = "bw_document"
    COLOR_DOCUMENT = "color_document"
    PHOTO = "photo"
    FLAT_BACKGROUND = "flat_background"


# Константи для порогів застосування корекцій
EPSILON = 0.001  # Поріг для ігнорування дуже малих значень

# Фіксований порядок кроків обробки (для пресетів)
PIPELINE_STEPS_FIXED_ORDER = [
    ("perspective",      "Авто-перспектива"),
    ("shadow_remove",    "Видалення тіней"),
    ("color_cast",       "Нейтралізація відтінку"),
    ("brightness",       "Авто-яскравість"),
    ("contrast",         "Авто-контраст"),
    ("hdr",              "HDR"),
    ("sharpen",          "Різкість"),
    ("grayscale",        "Grayscale / бінаризація"),
    ("white_background", "Білий фон"),
]


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


def decide_shadow_remove(
    doc_type: str,
    capture_condition: str,
    background_uniformity: float,
    face_detected: bool,
    had_color_cast: bool,
    settings: Optional[dict] = None,
) -> tuple[bool, str]:
    """Повертає рішення та причину для автоматичного видалення тіней.

    Функція не змінює зображення й не викликає алгоритм обробки. Вона є
    єдиним місцем для правил `doc_type × capture_condition × uniformity`.
    """
    settings = settings or {}
    low = settings.get("shadow_uniformity_low", 0.30)
    high = settings.get("shadow_uniformity_high", 0.55)

    if capture_condition == CAPTURE_SCREEN:
        return False, "screen_capture"

    if doc_type == DocType.PHOTO.value:
        photo_high = settings.get("shadow_uniformity_photo_high", 0.65)
        if capture_condition == CAPTURE_PHONE:
            photo_high *= 0.85
        if background_uniformity > photo_high:
            if face_detected:
                return False, f"photo face_detected (uniformity={background_uniformity:.2f})"
            return True, f"photo uniformity={background_uniformity:.2f}>{photo_high:.2f}"
        if background_uniformity < low:
            return False, f"photo low uniformity={background_uniformity:.2f}"
        return False, f"photo mid uniformity={background_uniformity:.2f}"

    if doc_type == DocType.COLOR_DOCUMENT.value:
        if background_uniformity > high:
            return True, f"color_doc uniformity={background_uniformity:.2f}>{high:.2f}"
        return False, f"color_doc low uniformity={background_uniformity:.2f}"

    if background_uniformity > high:
        if face_detected:
            return False, f"face_detected (uniformity={background_uniformity:.2f})"
        return True, f"uniformity={background_uniformity:.2f}>{high:.2f}"

    if background_uniformity < low:
        return False, f"uniformity={background_uniformity:.2f}<{low:.2f}"

    if doc_type == DocType.BW_DOCUMENT.value:
        if had_color_cast:
            return True, "bw_document + color_cast"
        if face_detected:
            return False, "face_detected (mid uniformity)"
        return True, f"doc_type={doc_type}"

    return False, ""
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
    settings: Optional[dict] = None,
) -> tuple[np.ndarray, str, list[dict]]:
    """
    Повний автоматичний pipeline з авто-визначенням типу документа.
    Повертає (результат, статус_повідомлення, список_кроків).

    Кожен елемент списку_кроків — dict:
        {"step": str, "applied": bool, "detail": str}

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
    settings — словник налаштувань для параметрів видалення тіней.
    """
    result = image.copy()
    log_entries: list[dict] = []

    # Діагностика фону — до будь-якої обробки (Задача 3)
    from processing import diagnostics as _diag
    _bg_uniformity, _detail_density = _diag.measure_background_metrics(image)

    # Класифікація умов зйомки (Задача 5)
    from processing.doc_classifier import classify_capture_conditions
    _capture_cond = classify_capture_conditions(image, background_uniformity=_bg_uniformity)

    # Записуємо початкову однорідність фону в статус
    log_entries.append({"step": "uniformity", "applied": True,
                        "detail": f"фон={_bg_uniformity:.2f}"})

    # Визначаємо список увімкнених кроків згідно з пресетом
    _preset_steps_map = {
        "doc_bw":    ["perspective", "shadow_remove", "brightness", "contrast", "sharpen", "grayscale", "white_background"],
        "doc_color": ["perspective", "shadow_remove", "brightness", "contrast", "sharpen", "white_background"],
        "photo":     ["perspective", "hdr", "sharpen"],
        "geometry":  ["perspective"],
    }
    _steps_enabled: list[str] | None = None
    if settings:
        _preset = settings.get("pipeline_preset", "doc_bw")
        if _preset != "custom":
            _steps_enabled = _preset_steps_map.get(_preset, [])
        else:
            _enabled_str = settings.get("pipeline_steps_enabled", "")
            _steps_enabled = [k.strip() for k in _enabled_str.split(",") if k.strip()] if _enabled_str else None

    # ── Детекція кутів перспективи ДО будь-якої обробки ──────────────────
    # Причина: тінь підкреслює межі документа → контраст країв вищий на оригіналі.
    # Після shadow_remove фон стає рівномірним і краї "губляться".
    _pre_detected_corners: np.ndarray | None = None
    _perspective_step_enabled = (
        _steps_enabled is None or "perspective" in (_steps_enabled or [])
    )
    _use_persp_flag = use_perspective or (_capture_cond == CAPTURE_SCREEN)

    if _perspective_step_enabled and _use_persp_flag:
        _pre_detected_corners = perspective.auto_detect_corners(image)
        if _pre_detected_corners is not None:
            log_entries.append({
                "step": "corners_pre_detected",
                "applied": True,
                "detail": "кути знайдено до обробки",
            })
            logger.debug("run_autofix: кути детектовано ДО обробки")
        else:
            logger.debug("run_autofix: кути не знайдено ДО обробки, спробуємо після")

    # Завдання 4.1: color_cast ДО класифікації документа
    # (після детекції кутів, але до класифікації)
    _had_color_cast = False
    result, _had_color_cast = color_cast.correct_color_cast(result)
    if _had_color_cast:
        log_entries.append({"step": "color_cast_pre", "applied": True,
                            "detail": "відтінок нейтралізовано до класифікації"})

    # Спочатку визначаємо тип документа (до будь-якої обробки!)
    if doc_type is None:
        doc_type = doc_classifier.classify(
            result,
            bw_std_thresh=classify_bw_std_thresh,
            edge_ratio_min=classify_edge_ratio_min,
            line_count_min=classify_line_count_min,
        )

    # Спеціальна обробка для flat_background: мінімальний набір кроків
    # (перспектива + білий фон), без shadow_remove, contrast, sharpen тощо.
    # Рівний фон без документа не потребує повної обробки — це запобігає
    # псуванню сканів, які помилково класифіковані як flat_background.
    if doc_type == DocType.FLAT_BACKGROUND.value:
        _steps_enabled = ["perspective", "white_background"]

    # Параметри shadow_remove
    _shadow_coarse_blend = settings.get("shadow_coarse_blend_color", 0.0) if settings else 0.0
    _shadow_detect_threshold = settings.get("shadow_detect_threshold", 80.0) if settings else 80.0
    _shadow_detect_ratio = settings.get("shadow_detect_ratio", 0.3) if settings else 0.3
    _shadow_is_color = (doc_type == DocType.COLOR_DOCUMENT.value)
    _shadow_bgr_mode = settings.get("shadow_bgr_mode", False) if settings else False

    # Завдання 1.6: detect_face один раз на початку, щоб не викликати до 3 разів
    from processing import diagnostics as _diag_face_once
    _face_detected: bool | None = None
    if doc_type != DocType.PHOTO.value:
        # Фото перевіряється окремо всередині циклу
        _face_detected = _diag_face_once.detect_face(result)

    # Проходимо циклом по фіксованому порядку кроків
    for step_key, _ in PIPELINE_STEPS_FIXED_ORDER:
        # Якщо steps_enabled визначено і крок не в списку — пропускаємо
        if _steps_enabled is not None and step_key not in _steps_enabled:
            continue

        elif step_key == "shadow_remove":
            shadow_mode = settings.get("shadow_remove_mode", "auto") if settings else "auto"

            if shadow_mode == "always":
                # Примусово — для будь-якого типу документа
                result = shadow_remove.remove_shadow(
                    result,
                    is_color_document=_shadow_is_color,
                    coarse_blend=_shadow_coarse_blend,
                    bgr_mode=_shadow_bgr_mode,
                )
                log_entries.append({"step": "shadow_remove", "applied": True,
                                    "detail": "тіні видалено (примусово)"})
            elif shadow_mode == "never":
                pass  # нічого не робимо
            else:  # auto — з урахуванням background_uniformity, doc_type та capture_conditions
                if doc_type == DocType.PHOTO.value:
                    photo_high = settings.get("shadow_uniformity_photo_high", 0.65) if settings else 0.65
                    if _capture_cond == CAPTURE_PHONE:
                        photo_high *= 0.85
                    photo_face_detected = (
                        _diag_face_once.detect_face(result)
                        if _bg_uniformity > photo_high else False
                    )
                else:
                    photo_face_detected = bool(_face_detected)
                _should_run, _reason = decide_shadow_remove(
                    doc_type,
                    _capture_cond,
                    _bg_uniformity,
                    photo_face_detected,
                    _had_color_cast,
                    settings,
                )
                if _should_run:
                    result, had_shadow = shadow_remove.auto_remove_shadow(
                        result,
                        is_color_document=_shadow_is_color,
                        coarse_blend=_shadow_coarse_blend,
                        detect_threshold=_shadow_detect_threshold,
                        detect_ratio=_shadow_detect_ratio,
                        bgr_mode=_shadow_bgr_mode,
                        background_uniformity=_bg_uniformity,
                    )
                    if had_shadow:
                        log_entries.append({"step": "shadow_remove", "applied": True,
                                           "detail": f"тіні видалено ({_reason})"})
                    # Нейтралізація кольорового відтінку для phone_camera
                    if had_shadow and _capture_cond == CAPTURE_PHONE:
                        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                        l_ch, a_ch, b_ch = cv2.split(lab)
                        bg_mask = l_ch > 180
                        bg_pixel_count = np.count_nonzero(bg_mask)
                        if bg_pixel_count > 0:
                            # Зсув a-каналу до нейтрального (128) на 30%
                            a_bg = a_ch[bg_mask]
                            b_bg = b_ch[bg_mask]
                            # посилено з 0.3 до 0.7: відтінок телефонної камери потребує сильнішої корекції
                            a_shift = (128.0 - float(np.median(a_bg.astype(np.float64)))) * 0.7
                            b_shift = (128.0 - float(np.median(b_bg.astype(np.float64)))) * 0.7
                            a_ch = np.clip(a_ch.astype(np.float32) + a_shift, 0, 255).astype(np.uint8)
                            b_ch = np.clip(b_ch.astype(np.float32) + b_shift, 0, 255).astype(np.uint8)
                            merged = cv2.merge([l_ch, a_ch, b_ch])
                            result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                            log_entries.append({"step": "color_neutralize", "applied": True,
                                               "detail": "нейтралізація відтінку"})
        elif step_key == "color_cast":
            # --- Корекція кольорового відтінку фону ---
            # Виконується після shadow_remove, бо тіні можуть спотворювати аналіз кольору
            if doc_type in (DocType.COLOR_DOCUMENT.value, DocType.BW_DOCUMENT.value):
                result, had_cast = color_cast.correct_color_cast(result)
                if had_cast:
                    log_entries.append({"step": "color_cast", "applied": True,
                                        "detail": "відтінок нейтралізовано"})

            # Висвітлення тіней — додаткове підсвічування (залишається як є)
            if shadow_highlight_strength > EPSILON:
                result = shadow_highlight.apply_shadow_highlight(
                    result, strength=shadow_highlight_strength
                )
                log_entries.append({"step": "shadow_highlight", "applied": True,
                                   "detail": f"підсвічування {shadow_highlight_strength:.2f}"})

        elif step_key == "perspective":
            _use_persp = use_perspective or _capture_cond == CAPTURE_SCREEN
            if _use_persp:
                if _pre_detected_corners is not None:
                    # Використовуємо кути знайдені ДО обробки
                    logger.debug("run_autofix: застосовуємо кути знайдені ДО обробки")
                    skewed = perspective.detect_skewed_sides(_pre_detected_corners)
                    has_skewed = any(skewed.values())
                    if has_skewed:
                        result = perspective.apply_correction(result, _pre_detected_corners)
                        # deskew після warp
                        angle = deskew_module.measure_skew_angle(result)
                        if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                            result = deskew_module.apply_deskew(result, angle)
                        _bg_uniformity, _detail_density = _diag.measure_background_metrics(result)
                        log_entries.append({
                            "step": "perspective",
                            "applied": True,
                            "detail": "перспектива виправлена (pre-detected) + deskew",
                        })
                    else:
                        # Тільки deskew
                        angle = deskew_module.measure_skew_angle(result)
                        if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                            result = deskew_module.apply_deskew(result, angle)
                            log_entries.append({
                                "step": "perspective",
                                "applied": True,
                                "detail": "deskew (pre-detected, нахил виправлено)",
                            })
                    # Скидаємо pre-detected щоб не застосувати двічі
                    _pre_detected_corners = None
                else:
                    # Fallback: стара логіка якщо pre-detection не знайшла кутів
                    result, persp_status = run_perspective_auto_smart(result, settings)
                    _bg_uniformity, _detail_density = _diag.measure_background_metrics(result)
                    if persp_status not in (
                        "перспектива не потрібна",
                        "перспектива не потребує корекції",
                    ):
                        log_entries.append({
                            "step": "perspective",
                            "applied": True,
                            "detail": persp_status,
                        })
                # Записуємо оновлену однорідність після перспективи
                log_entries.append({"step": "uniformity_updated", "applied": True,
                                    "detail": f"фон={_bg_uniformity:.2f} (після перспективи)"})

        elif step_key == "brightness":
            # Яскравість окремо не застосовується в autofix — вона в ручних налаштуваннях
            pass

        elif step_key == "contrast":
            result = run_contrast_advanced(result, autofix_contrast, contrast_mode)
            log_entries.append({"step": "contrast", "applied": True, "detail": f"контраст {autofix_contrast:.2f}"})

        elif step_key == "hdr":
            if doc_type == DocType.PHOTO.value:
                _use_hdr = settings.get("hdr_in_autofix", True) if settings else use_hdr
                if _use_hdr:
                    result = autofix.apply(result, sharpen_strength=0.0, hdr_strength=hdr_strength, use_hdr=True, adaptive_hdr=adaptive_hdr)
                    log_entries.append({"step": "hdr", "applied": True, "detail": "HDR"})
            # Для не-photo HDR пропускаємо

        elif step_key == "sharpen":
            # Різкість застосовується згідно з типом документа
            if doc_type == DocType.BW_DOCUMENT.value:
                # Завдання 1.5: тільки sharpen, без auto_contrast+grayscale всередині
                # (contrast і grayscale виконуються окремими кроками пізніше)
                result = sharpen.apply(result, strength=sharpen_strength)
                log_entries.append({"step": "sharpen", "applied": True, "detail": f"різкість {sharpen_strength:.2f}"})
                if bw_binary:
                    log_entries.append({"step": "binary", "applied": True, "detail": "бінаризація"})
            elif doc_type == DocType.COLOR_DOCUMENT.value:
                result = autofix.apply_color_document(result, sharpen_strength=sharpen_strength)
                log_entries.append({"step": "sharpen", "applied": True, "detail": f"різкість {sharpen_strength:.2f}"})
            else:
                result = sharpen.apply(result, strength=sharpen_strength)
                log_entries.append({"step": "sharpen", "applied": True, "detail": f"різкість {sharpen_strength:.2f}"})

        elif step_key == "grayscale":
            if doc_type == DocType.BW_DOCUMENT.value:
                # Для bw_document — конвертуємо в чб
                result = bc.to_grayscale(result)
                log_entries.append({"step": "grayscale", "applied": True, "detail": "ч-б"})

        elif step_key == "white_background":
            result, had_white = _apply_auto_white_background(result, doc_type)
            if had_white:
                log_entries.append({"step": "white_background", "applied": True, "detail": "білий фон"})

    # Додаємо інформацію про тип документа в лог (з capture_cond)
    if doc_type == DocType.BW_DOCUMENT.value:
        log_entries.append({"step": "doc_type", "applied": True, "detail": f"ч-б документ ({_capture_cond})"})
    elif doc_type == DocType.COLOR_DOCUMENT.value:
        log_entries.append({"step": "doc_type", "applied": True, "detail": f"кольоровий документ ({_capture_cond})"})
    else:
        log_entries.append({"step": "doc_type", "applied": True, "detail": f"фото ({_capture_cond})"})

    # Формат виходу: якщо не "auto" — примусово конвертуємо
    if output_color_mode == "grayscale":
        result = bc.to_grayscale(result)
        log_entries.append({"step": "color_mode", "applied": True, "detail": "ч-б"})
    elif output_color_mode == "binary":
        result = bc.to_grayscale(result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        tt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        result = cv2.cvtColor(tt, cv2.COLOR_GRAY2BGR)
        log_entries.append({"step": "color_mode", "applied": True, "detail": "бінаризація"})
    elif output_color_mode == "color":
        pass

    status_parts = [e["detail"] for e in log_entries if e["applied"]]
    status_msg = "Auto Fix: " + ", ".join(status_parts)
    return result, status_msg, log_entries


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


def run_perspective_auto_smart(
    image: np.ndarray,
    settings: dict | None = None,
    filename: str | None = None,
) -> tuple[np.ndarray, str]:
    """
    Розумна авто-перспектива з deskew та ітеративним уточненням.
    """
    corners = perspective.auto_detect_corners(image, filename=filename)

    if corners is not None:
        skewed = perspective.detect_skewed_sides(corners)
        has_skewed = any(skewed.values())

        if has_skewed:
            # Ітеративна корекція (до 2 проходів)
            _partial_val = settings.get("partial_perspective", False) if settings else False
            result, passes, final_skew = perspective.auto_correct_iterative(image, partial=_partial_val)
            # Deskew після останнього warp
            angle = deskew_module.measure_skew_angle(result)
            if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                result = deskew_module.apply_deskew(result, angle)
            status = f"перспектива виправлена ({passes} прохід(ів)) + deskew"
            return result, status
        else:
            angle = deskew_module.measure_skew_angle(image)
            if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                result = deskew_module.apply_deskew(image, angle)
                return result, "deskew (нахил виправлено)"
            else:
                return image.copy(), "перспектива не потребує корекції"
    else:
        angle = deskew_module.measure_skew_angle(image)
        if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
            result = deskew_module.apply_deskew(image, angle)
            return result, "deskew (нахил виправлено)"
        else:
            return image.copy(), "перспектива не потрібна"


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


def run_crop_pin_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp для кружечків перспективи без зміни розміру crop-результату."""
    return perspective.apply_crop_pin(image, corners)


def run_crop_rect(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Виконує просте осьовирівняне кадрування без padding і warp."""
    return perspective.apply_axis_aligned_crop(image, corners)


def detect_document_bounds(image: np.ndarray) -> np.ndarray | None:
    """Повертає осьову рамку документа без авто-перспективного warp."""
    return perspective.detect_document_bounds(image)


def detect_corners(image: np.ndarray, filename: str | None = None) -> np.ndarray | None:
    """
    Повертає 4 кути документа (для відображення у GUI)
    або None якщо не знайдено.
    """
    return perspective.auto_detect_corners(image, filename=filename)


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


def run_shadow_remove_manual(
    image: np.ndarray,
    is_color_document: bool,
    settings: Optional[dict] = None,
) -> tuple[np.ndarray, bool]:
    """Виконує лише примусове видалення тіні для окремої дії в GUI.

    Застосовано Варіант A: явне натискання кнопки завжди викликає
    ``remove_shadow`` і не запускає детекцію тіні та інші кроки pipeline.
    На результат впливають класифікація кольорового документа, значення
    ``shadow_coarse_blend_color`` і ``shadow_bgr_mode`` з налаштувань.
    Параметри детекції та однорідності фону тут навмисно не використовуються.
    """
    settings = settings or {}
    coarse_blend = settings.get("shadow_coarse_blend_color", 0.0)
    bgr_mode = settings.get("shadow_bgr_mode", False)
    result = shadow_remove.remove_shadow(
        image,
        is_color_document=is_color_document,
        coarse_blend=coarse_blend,
        bgr_mode=bgr_mode,
    )
    # Варіант A є форсованою дією, тому ручний запуск завжди вважається
    # застосованим, навіть якщо в зображенні фактично не було тіні.
    return result, True


def run_universal(
    image: np.ndarray,
    settings: Optional[dict] = None,
    doc_type: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """Застосувати налаштований набір кроків однією операцією.

    Кроки виконуються тільки якщо їхні ``universal_*_enabled`` прапорці
    увімкнені, але завжди в канонічному порядку ``PIPELINE_STEPS_FIXED_ORDER``.
    Перспектива навмисно виключена: її редагування має окремий UI-режим.
    Видалення тіні використовує той самий форсований Варіант A, що й окрема
    кнопка «Прибрати тінь».
    """
    settings = dict(settings or {})
    enabled_steps = [
        step_key
        for step_key, _ in PIPELINE_STEPS_FIXED_ORDER
        if step_key != "perspective"
        and settings.get(f"universal_{step_key}_enabled", False)
    ]
    if not enabled_steps:
        return image.copy(), "Не обрано жодного кроку"

    result = image.copy()
    log_entries: list[str] = []

    # Класифікація потрібна тільки крокам, що використовують тип документа.
    normalized_doc_type = getattr(doc_type, "value", doc_type)
    if normalized_doc_type is None and (
        "shadow_remove" in enabled_steps or "white_background" in enabled_steps
    ):
        normalized_doc_type = run_classify(
            image,
            bw_std_thresh=settings.get("classify_bw_std_thresh", 20.0),
            edge_ratio_min=settings.get("classify_edge_ratio_min", 0.03),
            line_count_min=settings.get("classify_line_count_min", 3),
        )
    is_color_document = normalized_doc_type == DocType.COLOR_DOCUMENT.value

    for step_key in enabled_steps:
        if step_key == "shadow_remove":
            result, _ = run_shadow_remove_manual(
                result,
                is_color_document=is_color_document,
                settings=settings,
            )
            log_entries.append("тіні")

        elif step_key == "color_cast":
            result, changed = color_cast.correct_color_cast(result)
            log_entries.append("нейтралізація відтінку" if changed else "нейтралізація відтінку (без змін)")

        elif step_key == "brightness":
            value = settings.get("universal_brightness_value", 0.0)
            result = run_brightness(result, value)
            log_entries.append(f"яскравість={value:.2f}")

        elif step_key == "contrast":
            value = settings.get("universal_contrast_value", 0.0)
            mode = settings.get("contrast_mode", "linear")
            result = run_contrast_advanced(result, value, mode)
            log_entries.append(f"контраст={value:.2f}")

        elif step_key == "hdr":
            value = settings.get("universal_hdr_value", 0.0)
            if value > EPSILON:
                result = hdr.apply(result, strength=value, manual_mode=True)
            log_entries.append(f"HDR={value:.2f}")

        elif step_key == "sharpen":
            value = settings.get("universal_sharpen_value", 0.4)
            result = sharpen.apply(result, strength=value)
            log_entries.append(f"різкість={value:.2f}")

        elif step_key == "grayscale":
            result = run_grayscale(result)
            log_entries.append("чорно-біле")

        elif step_key == "white_background":
            result, changed = _apply_auto_white_background(result, normalized_doc_type)
            log_entries.append("білий фон" if changed else "білий фон (без змін)")

    return result, "Універсальна: " + ", ".join(log_entries)


def _apply_auto_white_background(image: np.ndarray, doc_type: str) -> tuple[np.ndarray, bool]:
    """
    Відбілює фон документа, перераховуючи background_uniformity/
    detail_density на АКТУАЛЬНОМУ image (а не на застарілій діагностиці
    з початку pipeline). Спільна для run_autofix і run_full_auto, щоб
    обидва шляхи поводились однаково і щоб цю помилку не повторили
    в майбутньому в одному з двох місць.

    Не застосовується до фото.
    """
    if doc_type == DocType.PHOTO.value:
        return image, False
    from processing import diagnostics as diag_module
    uniformity, detail_density = diag_module.measure_background_metrics(image)
    return white_background.make_background_white(
        image,
        background_uniformity=uniformity,
        detail_density=detail_density,
    )




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
            result = hdr.apply_adaptive(result, strength=hdr_strength, text_mask=mask, manual_mode=True)
        else:
            result = hdr.apply(result, strength=hdr_strength, manual_mode=True)
    if sharpen_strength > EPSILON:
        result = sharpen.apply(result, strength=sharpen_strength)
    return result

def _compute_adaptive_params(diag) -> dict:
    """
    Obchyslyuye adaptyvni parametry obrobky na osnovi diagnostyky zobrazhennya.

    Povernaye dict z klyuchamy:
    - hdr_strength: syla HDR (0.2..0.8), zalezhyt vid detail_density
    - noise_floor: bazove 1.5, vyshche dlya shumnyh zobrazhen
    - contrast_strength: 0 pry dynamic_range > 180, inakshe proporcyno
    - shadow_remove: True yakshcho gradient + uniformity > 0.4 + document
    - brightness_needed: True yakshcho middle L < 80
    - brightness_correction: znachennya z diagnostyky yakshcho needed
    - sharpen_strength: z diagnostyky (blur_sharpen_strength)
    """
    hdr_strength = min(0.8, max(0.2, diag.detail_density * 0.8))

    noise_floor = max(1.5, diag.noise_level)

    if diag.dynamic_range > 180:
        contrast_strength = 0.0
    else:
        contrast_strength = (180.0 - diag.dynamic_range) / 180.0
    contrast_strength = max(0.0, min(1.0, contrast_strength))

    shadow_remove = (
        diag.gradient_has
        and diag.background_uniformity > 0.4
        and diag.doc_type in ("bw_document", "color_document")
    )

    brightness_needed = diag.brightness_mean_l < 80
    brightness_correction = diag.brightness_correction if brightness_needed else 0.0

    sharpen_strength = diag.blur_sharpen_strength

    return {
        "hdr_strength": hdr_strength,
        "noise_floor": noise_floor,
        "contrast_strength": contrast_strength,
        "shadow_remove": shadow_remove,
        "brightness_needed": brightness_needed,
        "brightness_correction": brightness_correction,
        "sharpen_strength": sharpen_strength,
    }



def run_full_auto(
    image: np.ndarray,
    settings: dict,
    dry_run: bool = False,
) -> tuple[np.ndarray, str, dict]:
    """
    Full Auto pipeline z adaptyvnymy parametramy.

    1. Diagnostyka zobrazhennya
    2. Obchyslennya adaptyvnyh parametriv
    3. Vyklyk run_autofix z tsymy parametramy
    4. Spetsialna obrobka flat_background

    Povernaye (result, status_msg, steps_dict).
    """
    if dry_run:
        return image.copy(), "Full Auto: dry run (bez obrobky)", {}

    from processing.diagnostics import diagnose
    diag = diagnose(image, settings)
    params = _compute_adaptive_params(diag)

    use_persp = settings.get("full_auto_perspective", False)
    hdr_strength = settings.get("hdr_strength", 0.5)
    sharpen_strength = settings.get("sharpen_strength", 0.4)

    # Build modified settings to exclude perspective when full_auto_perspective=False
    mod_settings = dict(settings) if settings else {}
    all_steps = "shadow_remove,color_cast,brightness,contrast,hdr,sharpen,grayscale,white_background"
    if use_persp:
        all_steps = "perspective," + all_steps
    mod_settings["pipeline_preset"] = "custom"
    mod_settings["pipeline_steps_enabled"] = all_steps

    result, autofix_status, log_entries = run_autofix(
        image,
        doc_type=diag.doc_type,
        use_perspective=use_persp,
        hdr_strength=hdr_strength,
        sharpen_strength=sharpen_strength,
        use_hdr=settings.get("hdr_in_autofix", True),
        bw_binary=settings.get("bw_binary", False),
        classify_bw_std_thresh=settings.get("bw_std_thresh", 20.0),
        classify_edge_ratio_min=settings.get("edge_ratio_min", 0.03),
        classify_line_count_min=settings.get("line_count_min", 3),
        shadow_highlight_strength=settings.get("shadow_highlight_strength", 0.0),
        output_color_mode=settings.get("output_color_mode", "auto"),
        autofix_contrast=settings.get("autofix_contrast", 0.15),
        contrast_mode=settings.get("contrast_mode", "linear"),
        settings=mod_settings,
    )

    if diag.doc_type == "flat_background":
        status = f"Full Auto: rivnyi fon, {autofix_status}"
    else:
        status = f"Full Auto: {autofix_status}"

    steps_dict = {"doc_type": diag.doc_type, "adaptive_params": params}

    return result, status, steps_dict
