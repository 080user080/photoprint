"""
Pipeline — єдина точка входу для GUI.
GUI викликає тільки pipeline, не знає про внутрішні модулі.
"""

import logging
from enum import Enum
from typing import Optional
import cv2
import numpy as np
from processing import autofix, sharpen, hdr, perspective, brightness_contrast as bc, doc_classifier, shadow_highlight, shadow_remove, white_background, deskew as deskew_module

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
    from processing.doc_classifier import classify_capture_conditions, CAPTURE_SCREEN, CAPTURE_PHONE
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

    # Спочатку визначаємо тип документа (до будь-якої обробки!)
    if doc_type is None:
        doc_type = doc_classifier.classify(
            result,
            bw_std_thresh=classify_bw_std_thresh,
            edge_ratio_min=classify_edge_ratio_min,
            line_count_min=classify_line_count_min,
        )

    # Параметри shadow_remove
    _shadow_coarse_blend = settings.get("shadow_coarse_blend_color", 0.0) if settings else 0.0
    _shadow_detect_threshold = settings.get("shadow_detect_threshold", 80.0) if settings else 80.0
    _shadow_detect_ratio = settings.get("shadow_detect_ratio", 0.3) if settings else 0.3
    _shadow_is_color = (doc_type == DocType.COLOR_DOCUMENT.value)
    _shadow_bgr_mode = settings.get("shadow_bgr_mode", False) if settings else False
    _shadow_unif_low = settings.get("shadow_uniformity_low", 0.30) if settings else 0.30
    _shadow_unif_high = settings.get("shadow_uniformity_high", 0.55) if settings else 0.55

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
                _should_run = False
                _reason = ""

                if _capture_cond == CAPTURE_SCREEN:
                    # screen_capture — не запускаємо (екран рівномірний, не тінь)
                    _should_run = False
                    _reason = "screen_capture"
                elif doc_type == DocType.PHOTO.value:
                    # Фото — запускаємо ТІЛЬКИ при високій однорідності фону.
                    # Фото документа на білому фоні (зроблене на телефон) має високу uniformity.
                    # Пейзажі, портрети — низьку uniformity, тому автоматично захищені.
                    _shadow_unif_high_photo = settings.get("shadow_uniformity_photo_high", 0.65) if settings else 0.65
                    if _bg_uniformity > _shadow_unif_high_photo:
                        # Додаткова перевірка: якщо є обличчя — не видаляємо тіні
                        from processing import diagnostics as _diag_face_photo
                        if _diag_face_photo.detect_face(result):
                            _should_run = False
                            _reason = f"photo face_detected (uniformity={_bg_uniformity:.2f})"
                        else:
                            _should_run = True
                            _reason = f"photo uniformity={_bg_uniformity:.2f}>{_shadow_unif_high_photo:.2f}"
                    elif _bg_uniformity < _shadow_unif_low:
                        # Складний фон — не запускаємо
                        _should_run = False
                        _reason = f"photo low uniformity={_bg_uniformity:.2f}"
                    else:
                        # Проміжний діапазон — фото ризиковані, краще консервативно
                        _should_run = False
                        _reason = f"photo mid uniformity={_bg_uniformity:.2f}"
                elif doc_type == DocType.COLOR_DOCUMENT.value:
                    # Кольоровий документ — запускаємо ТІЛЬКИ при високій однорідності фону,
                    # інакше ризикуємо зіпсувати кольоровий фон/зображення на документі.
                    # Паспорти, посвідчення з кольоровим фоном псуються shadow_remove,
                    # але якщо фон реально рівномірний (білий/однотонний) — тіні треба прибрати.
                    if _bg_uniformity > _shadow_unif_high:
                        _should_run = True
                        _reason = f"color_doc uniformity={_bg_uniformity:.2f}>{_shadow_unif_high:.2f}"
                    else:
                        _should_run = False
                        _reason = f"color_doc low uniformity={_bg_uniformity:.2f}"
                elif _bg_uniformity > _shadow_unif_high:
                    # Однорідний фон bw_document / flat_background.
                    # Додаткова перевірка: якщо є обличчя — це документ з портретом,
                    # shadow_remove зіпсує фото особи.
                    from processing import diagnostics as _diag_face
                    if _diag_face.detect_face(result):
                        _should_run = False
                        _reason = f"face_detected (uniformity={_bg_uniformity:.2f})"
                    else:
                        _should_run = True
                        _reason = f"uniformity={_bg_uniformity:.2f}>{_shadow_unif_high:.2f}"
                elif _bg_uniformity < _shadow_unif_low:
                    # Складний фон — не запускаємо
                    _should_run = False
                    _reason = f"uniformity={_bg_uniformity:.2f}<{_shadow_unif_low:.2f}"
                else:
                    # Проміжний діапазон — тільки для bw_document,
                    # і тільки якщо немає обличчя
                    if doc_type == DocType.BW_DOCUMENT.value:
                        from processing import diagnostics as _diag_face
                        if _diag_face.detect_face(result):
                            _should_run = False
                            _reason = "face_detected (mid uniformity)"
                        else:
                            _should_run = True
                            _reason = f"doc_type={doc_type}"
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
                            a_shift = (128.0 - float(np.median(a_bg))) * 0.3
                            b_shift = (128.0 - float(np.median(b_bg))) * 0.3
                            a_ch = np.clip(a_ch.astype(np.float32) + a_shift, 0, 255).astype(np.uint8)
                            b_ch = np.clip(b_ch.astype(np.float32) + b_shift, 0, 255).astype(np.uint8)
                            merged = cv2.merge([l_ch, a_ch, b_ch])
                            result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                            log_entries.append({"step": "color_neutralize", "applied": True,
                                               "detail": "нейтралізація відтінку"})
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
                result = autofix.apply_bw_document(result, sharpen_strength=sharpen_strength, binary=bw_binary)
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
) -> tuple[np.ndarray, str]:
    """
    Розумна авто-перспектива з deskew та ітеративним уточненням.
    """
    corners = perspective.auto_detect_corners(image)

    if corners is not None:
        skewed = perspective.detect_skewed_sides(corners)
        has_skewed = any(skewed.values())

        if has_skewed:
            # Ітеративна корекція (до 2 проходів)
            result, passes, final_skew = perspective.auto_correct_iterative(image)
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