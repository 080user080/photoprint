"""
Модуль діагностики зображення.
Аналізує градієнт, контраст, яскравість, розмиття, перспективу та тип документа.
Не змінює вхідне зображення.
"""

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

# Константи для діагностики
DIAGNOSTICS_RESIZE_MAX = 600       # розмір для швидкої діагностики

GRADIENT_L_DIFF_THRESHOLD = 30     # мінімальна різниця медіан для "є градієнт"
GRADIENT_L_MAX_EXPECTED = 120      # "максимальний" очікуваний діапазон (для нормалізації)

CONTRAST_RANGE_THRESHOLD = 160     # якщо p99-p1 < цього — контраст низький
CONTRAST_RANGE_MAX = 254           # максимально можливий діапазон

BLUR_THRESHOLD = 80.0              # збіг з autosharp_threshold в налаштуваннях
BLUR_MAX_VARIANCE = 500.0          # вище цього — "достатньо різке"

DARK_MEAN_THRESHOLD = 80           # середнє L нижче цього — "занадто темне"
BRIGHT_MEAN_THRESHOLD = 200        # середнє L вище цього — "занадто світле"
OVEREXPOSED_L_THRESHOLD = 250      # пікселі вище цього вважаються пересвіченими
UNDEREXPOSED_L_THRESHOLD = 5       # пікселі нижче цього вважаються недосвіченими

PERSPECTIVE_SKEW_THRESHOLD = 0.03  # 3% від розміру зображення — мінімальне викривлення


@dataclass
class DiagnosticResult:
    """Результат повної діагностики зображення."""
    doc_type: str                    # "bw_document" / "color_document" / "photo"

    # Градієнт фону
    gradient_has: bool
    gradient_strength: float         # 0..1, нормалізована сила
    gradient_direction: str          # "horizontal" / "vertical" / "both" / "none"

    # Контраст
    contrast_range_l: float          # p99 - p1 каналу L
    contrast_strength_needed: float  # 0..1, наскільки потрібна корекція
    overexposed_ratio: float         # частка пікселів > 250
    underexposed_ratio: float        # частка пікселів < 5

    # Яскравість
    brightness_mean_l: float         # середнє L
    brightness_correction: float     # -1..1, відхилення від нейтрального

    # Розмиття
    blur_variance: float             # Laplacian variance
    blur_strength_needed: float      # 0..1, наскільки потрібна різкість
    blur_sharpen_strength: float     # конкретне значення сили різкості 0..1

    # Перспектива
    perspective_has: bool
    perspective_corners: np.ndarray | None   # shape (4,2) або None
    perspective_skew_ratio: float    # відносне відхилення від прямокутника


# ---------------------------------------------------------------------------
# Внутрішні функції (з підкресленням)
# ---------------------------------------------------------------------------

def _resize_for_analysis(image: np.ndarray, max_dim: int = DIAGNOSTICS_RESIZE_MAX) -> np.ndarray:
    """
    Зменшує зображення до max_dim по більшій стороні якщо більше.
    Повертає копію або оригінал якщо вже менше.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy()
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _measure_gradient(image: np.ndarray) -> tuple[bool, float, str]:
    """
    Вимірює градієнт фону.
    Повертає (has_gradient, strength, direction).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    h, w = l.shape
    # Розбиваємо на сітку 4×4
    blocks_h = max(1, h // 4)
    blocks_w = max(1, w // 4)

    medians = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            y_start = i * blocks_h
            y_end = min((i + 1) * blocks_h, h)
            x_start = j * blocks_w
            x_end = min((j + 1) * blocks_w, w)
            block = l[y_start:y_end, x_start:x_end]
            medians[i, j] = float(np.median(block))

    diff_total = float(np.max(medians) - np.min(medians))

    # Варіація по рядках і стовпцях
    row_means = np.mean(medians, axis=1)  # середнє по рядках
    col_means = np.mean(medians, axis=0)  # середнє по стовпцях
    std_rows = float(np.std(row_means))
    std_cols = float(np.std(col_means))

    has_gradient = diff_total > GRADIENT_L_DIFF_THRESHOLD
    strength = min(1.0, diff_total / GRADIENT_L_MAX_EXPECTED)

    # Напрямок
    if std_cols > std_rows * 1.5:
        direction = "horizontal"
    elif std_rows > std_cols * 1.5:
        direction = "vertical"
    elif has_gradient and std_cols > 0 and std_rows > 0:
        direction = "both"
    else:
        direction = "none"

    return has_gradient, strength, direction


def _measure_contrast(image: np.ndarray) -> tuple[float, float, float, float]:
    """
    Вимірює контраст.
    Повертає (range_l, strength_needed, overexposed_ratio, underexposed_ratio).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    p1 = float(np.percentile(l, 1))
    p99 = float(np.percentile(l, 99))
    range_l = p99 - p1

    # strength_needed: якщо діапазон вже широкий — 0
    strength_needed = max(0.0, (CONTRAST_RANGE_THRESHOLD - range_l) / CONTRAST_RANGE_THRESHOLD)
    strength_needed = min(1.0, strength_needed)

    overexposed_ratio = float(np.mean(l > OVEREXPOSED_L_THRESHOLD))
    underexposed_ratio = float(np.mean(l < UNDEREXPOSED_L_THRESHOLD))

    return range_l, strength_needed, overexposed_ratio, underexposed_ratio


def _measure_brightness(image: np.ndarray) -> tuple[float, float]:
    """
    Вимірює яскравість.
    Повертає (mean_l, correction).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    mean_l = float(np.mean(l))
    correction = (127.0 - mean_l) / 127.0
    correction = max(-1.0, min(1.0, correction))

    return mean_l, correction


def _measure_blur(image: np.ndarray, settings: dict) -> tuple[float, float, float]:
    """
    Вимірює розмиття.
    Повертає (variance, strength_needed, sharpen_strength).
    """
    from processing import sharpen

    variance = sharpen.measure_sharpness(image)

    max_sharpen = settings.get("autosharp_max_strength", 0.7)
    threshold = settings.get("autosharp_threshold", BLUR_THRESHOLD)

    if variance >= threshold:
        strength_needed = 0.0
        sharpen_strength = 0.0
    else:
        strength_needed = max(0.0, 1.0 - variance / threshold)
        sharpen_strength = max(0.15, strength_needed * max_sharpen)

    return variance, strength_needed, sharpen_strength


def _measure_perspective(image: np.ndarray) -> tuple[bool, np.ndarray | None, float]:
    """
    Вимірює перспективу.
    Не викликає _resize_for_analysis — auto_detect_corners має власний max_dim.
    Повертає (has_perspective, corners or None, skew_ratio).
    """
    from processing import perspective

    corners = perspective.auto_detect_corners(image)
    if corners is None:
        return False, None, 0.0

    ordered = perspective._order_points(corners)
    tl, tr, br, bl = ordered

    top_skew = abs(float(tl[1] - tr[1]))
    bottom_skew = abs(float(bl[1] - br[1]))
    left_skew = abs(float(tl[0] - bl[0]))
    right_skew = abs(float(tr[0] - br[0]))

    max_skew = max(top_skew, bottom_skew, left_skew, right_skew)
    h, w = image.shape[:2]
    skew_ratio = max_skew / max(h, w)

    has_perspective = skew_ratio > PERSPECTIVE_SKEW_THRESHOLD
    return has_perspective, corners if has_perspective else None, skew_ratio


def _measure_doc_type(image: np.ndarray, settings: dict) -> str:
    """
    Визначає тип документа.
    """
    from processing import doc_classifier

    bw_std_thresh = settings.get("bw_std_thresh", 20.0)
    edge_ratio_min = settings.get("edge_ratio_min", 0.03)
    line_count_min = settings.get("line_count_min", 3)

    return doc_classifier.classify(
        image,
        bw_std_thresh=bw_std_thresh,
        edge_ratio_min=edge_ratio_min,
        line_count_min=line_count_min,
    )


# ---------------------------------------------------------------------------
# Публічні функції
# ---------------------------------------------------------------------------

def diagnose(image: np.ndarray, settings: dict) -> DiagnosticResult:
    """
    Повна діагностика зображення.
    Вхідне зображення не змінюється.
    """
    # Зменшуємо для швидкого аналізу
    small = _resize_for_analysis(image, DIAGNOSTICS_RESIZE_MAX)

    # Градієнт
    gradient_has, gradient_strength, gradient_direction = _measure_gradient(small)

    # Контраст
    contrast_range_l, contrast_strength_needed, overexposed_ratio, underexposed_ratio = _measure_contrast(small)

    # Яскравість
    brightness_mean_l, brightness_correction = _measure_brightness(small)

    # Розмиття
    blur_variance, blur_strength_needed, blur_sharpen_strength = _measure_blur(small, settings)

    # Перспектива (на оригіналі!)
    perspective_has, perspective_corners, perspective_skew_ratio = _measure_perspective(image)

    # Тип документа
    doc_type = _measure_doc_type(small, settings)

    return DiagnosticResult(
        doc_type=doc_type,
        gradient_has=gradient_has,
        gradient_strength=gradient_strength,
        gradient_direction=gradient_direction,
        contrast_range_l=contrast_range_l,
        contrast_strength_needed=contrast_strength_needed,
        overexposed_ratio=overexposed_ratio,
        underexposed_ratio=underexposed_ratio,
        brightness_mean_l=brightness_mean_l,
        brightness_correction=brightness_correction,
        blur_variance=blur_variance,
        blur_strength_needed=blur_strength_needed,
        blur_sharpen_strength=blur_sharpen_strength,
        perspective_has=perspective_has,
        perspective_corners=perspective_corners,
        perspective_skew_ratio=perspective_skew_ratio,
    )


def partial_rediagnose(image: np.ndarray, settings: dict, fields: list[str]) -> dict:
    """
    Частковий перерахунок діагностики після проміжних кроків.
    Приймає список полів для перерахунку (["contrast", "brightness", "blur"]).
    Не перераховує перспективу і тип документа.
    Повертає словник з оновленими значеннями.
    """
    small = _resize_for_analysis(image, DIAGNOSTICS_RESIZE_MAX)
    result = {}

    if "contrast" in fields:
        range_l, strength_needed, overexposed_ratio, underexposed_ratio = _measure_contrast(small)
        result["contrast_range_l"] = range_l
        result["contrast_strength_needed"] = strength_needed
        result["overexposed_ratio"] = overexposed_ratio
        result["underexposed_ratio"] = underexposed_ratio

    if "brightness" in fields:
        mean_l, correction = _measure_brightness(small)
        result["brightness_mean_l"] = mean_l
        result["brightness_correction"] = correction

    if "blur" in fields:
        variance, strength_needed, sharpen_strength = _measure_blur(small, settings)
        result["blur_variance"] = variance
        result["blur_strength_needed"] = strength_needed
        result["blur_sharpen_strength"] = sharpen_strength

    return result