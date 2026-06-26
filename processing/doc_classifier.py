"""Класифікація типу зображення: bw_document / color_document / photo."""

from typing import Literal
import cv2
import numpy as np
import logging

# Типи документів для type hints
DocType = Literal["bw_document", "color_document", "photo", "flat_background"]

# Константи для Canny edge detection
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150

# Константи для Hough Lines
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH_RATIO = 5  # minLineLength = min(shape) // 5
HOUGH_MAX_LINE_GAP = 10

# Константи для ресайзу для аналізу
ANALYSIS_SCALE = 0.3

# Поріг "кольоровості" одного пікселя: |a-128| або |b-128| вище цього —
# пікселя вважаємо кольоровим (не нейтральним/сірим).
# Зменшено з 15.0 до 10.0 для кращого виявлення блідих кольорів.
CHROMA_PIXEL_THRESHOLD = 10.0

# Якщо частка кольорових пікселів (за CHROMA_PIXEL_THRESHOLD) перевищує
# цей порядок — зображення містить значущий кольоровий контент і НЕ може
# бути "bw_document", незалежно від глобального std_a/std_b.
# Зменшено з 0.02 до 0.01 для більш чутливого виявлення кольору.
COLOR_PIXEL_RATIO_MIN = 0.01

# Умови зйомки
SCREEN_EDGE_DARK_THRESHOLD = 40    # пікселі темніші за це — підозра на рамку UI
SCREEN_EDGE_STRIP_WIDTH = 15       # ширина смуги по краях для аналізу
SCREEN_EDGE_DARK_RATIO = 0.6       # якщо >60% пікселів смуги темні → screen_capture
PHONE_WARM_B_THRESHOLD = 133       # b-канал LAB > цього → теплий відтінок фону
PHONE_COOL_B_THRESHOLD = 123       # b-канал LAB < цього → холодний відтінок фону
FLAT_UNIFORM_BG_THRESHOLD = 0.60   # background_uniformity > цього → flat_uniform

CAPTURE_SCREEN = "screen_capture"
CAPTURE_PHONE = "phone_camera"
CAPTURE_FLAT = "flat_uniform"
CAPTURE_UNKNOWN = "unknown"

# Логгер для діагностики
_logger = logging.getLogger(__name__)


def _has_color_content(a_ch: np.ndarray, b_ch: np.ndarray) -> tuple[float, bool]:
    """
    Локальна перевірка наявності кольору, стійка до великого нейтрального фону.

    Повертає (color_pixel_ratio, has_color_content):
      color_pixel_ratio — частка пікселів, де |a-128| або |b-128| > CHROMA_PIXEL_THRESHOLD
      has_color_content — True, якщо color_pixel_ratio >= COLOR_PIXEL_RATIO_MIN
    """
    a_dev = np.abs(a_ch.astype(np.float32) - 128.0)
    b_dev = np.abs(b_ch.astype(np.float32) - 128.0)
    chroma = np.maximum(a_dev, b_dev)
    color_pixel_ratio = float(np.mean(chroma > CHROMA_PIXEL_THRESHOLD))
    has_color_content = color_pixel_ratio >= COLOR_PIXEL_RATIO_MIN
    return color_pixel_ratio, has_color_content


def _has_histogram_color_content(a_ch: np.ndarray, b_ch: np.ndarray) -> bool:
    """
    Гістограмна перевірка наявності кольору.
    Якщо гістограми a або b каналу мають значущі піки (≥0.5% пікселів)
    за межами діапазону [128 - CHROMA_PIXEL_THRESHOLD, 128 + CHROMA_PIXEL_THRESHOLD],
    то зображення містить колір.

    Це додаткова перевірка для виявлення кольору навіть при низькому глобальному std,
    коли колір займає малу площу (наприклад, кольоровий об'єкт на великому білому фоні).
    """
    low = 128 - int(CHROMA_PIXEL_THRESHOLD)
    high = 128 + int(CHROMA_PIXEL_THRESHOLD)
    
    # Рахуємо гістограму з 256 бінами
    hist_a = cv2.calcHist([a_ch], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b_ch], [0], None, [256], [0, 256]).flatten()
    
    # Пікселі за межами нейтрального діапазону
    out_of_range_a = np.sum(hist_a[:low]) + np.sum(hist_a[high+1:])
    out_of_range_b = np.sum(hist_b[:low]) + np.sum(hist_b[high+1:])
    
    total_pixels = a_ch.size
    ratio_a = out_of_range_a / total_pixels
    ratio_b = out_of_range_b / total_pixels
    
    # Якщо хоча б 0.1% пікселів виходять за межі нейтрального діапазону — є колір
    return ratio_a > 0.001 or ratio_b > 0.001


# Константи для виявлення flat_background (рівний фон)
FLAT_BG_UNIFORMITY_THRESH = 0.70    # >70% площі — рівний фон
FLAT_BG_DETAIL_DENSITY_THRESH = 0.15  # <15% деталей


def _local_std_map(gray: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Локальне стандартне відхилення через box filter."""
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(f * f, -1, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def _is_flat_background(small: np.ndarray) -> bool:
    """
    Перевіряє, чи є зображення рівним фоном (flat_background):
    - background_uniformity > 0.7 (більшість площі — рівна)
    - detail_density < 0.15 (мало деталей)
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    std_map = _local_std_map(gray)

    # Background uniformity: частка пікселів з низьким локальним std
    uniformity = float(np.mean(std_map < 8.0))

    # Detail density: середнє значення detail_mask
    ref_std = max(float(np.percentile(std_map, 90.0)), 3.0)
    detail_mask = np.clip(std_map / ref_std, 0.0, 1.0)
    detail_density = float(np.mean(detail_mask))

    return uniformity > FLAT_BG_UNIFORMITY_THRESH and detail_density < FLAT_BG_DETAIL_DENSITY_THRESH


def classify(
    image: np.ndarray,
    bw_std_thresh: float = 10.0,
    edge_ratio_min: float = 0.03,
    line_count_min: int = 3,
) -> DocType:
    """
    Повертає тип документа: 'bw_document' | 'color_document' | 'photo' | 'flat_background'.
    
    bw_std_thresh: поріг стандартного відхилення a/b каналів у LAB.
        Зменшено з 20.0 до 10.0 для запобігання помилковій класифікації
        кольорових зображень з низькою насиченістю як чорно-білих документів.
    
    flat_background: рівний фон без документа (скріншоти, фото стіни тощо).
    """
    small = cv2.resize(image, (0, 0), fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Перевірка на flat_background — виконується першою, найшвидша
    if _is_flat_background(small):
        _logger.debug("-> flat_background")
        return "flat_background"

    std_a = float(np.std(a_ch))
    std_b = float(np.std(b_ch))

    # Локальна перевірка наявності кольору (стійка до великого нейтрального фону)
    color_pixel_ratio, has_color_content = _has_color_content(a_ch, b_ch)
    
    # Гістограмна перевірка — додатковий захист від помилкової класифікації
    has_histogram_color = _has_histogram_color_content(a_ch, b_ch)

    edges = cv2.Canny(l_ch, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)

    min_line_length = min(small.shape[:2]) // HOUGH_MIN_LINE_LENGTH_RATIO
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
                            minLineLength=min_line_length, maxLineGap=HOUGH_MAX_LINE_GAP)
    line_count = len(lines) if lines is not None else 0

    # Логування для діагностики
    _logger.debug(
        "classify: std_a=%.2f, std_b=%.2f, color_pixel_ratio=%.4f, "
        "has_color_content=%s, has_histogram_color=%s, "
        "edge_ratio=%.4f, line_count=%d, bw_std_thresh=%.1f",
        std_a, std_b, color_pixel_ratio,
        has_color_content, has_histogram_color,
        edge_ratio, line_count, bw_std_thresh
    )

    # Чорно-білий: глобально низький std a/b І жодного значущого локального кольору
    # І жодного гістограмного кольору
    is_achromatic = (
        std_a < bw_std_thresh and std_b < bw_std_thresh
        and not has_color_content
        and not has_histogram_color
    )

    if is_achromatic:
        if edge_ratio >= edge_ratio_min and line_count >= line_count_min:
            _logger.debug("-> bw_document")
            return "bw_document"
        _logger.debug("-> photo (achromatic, no edges)")
        return "photo"

    # Кольоровий — документ чи фото
    if edge_ratio >= edge_ratio_min and line_count >= line_count_min:
        _logger.debug("-> color_document")
        return "color_document"
    _logger.debug("-> photo (color)")
    return "photo"


# ---------------------------------------------------------------------------
# Capture conditions classification (Задача 4)
# ---------------------------------------------------------------------------


def _detect_screen_capture(small: np.ndarray) -> bool:
    """
    Визначає, чи є зображення захопленням екрану (screen capture).
    Аналізує краї зображення на наявність темних смуг (рамка UI/вікна).

    small: зменшене зображення BGR (до ~300px по більшій стороні).
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sw = min(SCREEN_EDGE_STRIP_WIDTH, w // 4)
    sh = min(SCREEN_EDGE_STRIP_WIDTH, h // 4)
    if sw < 2 or sh < 2:
        return False

    # Верхня, нижня, ліва, права смуги
    strips = [
        gray[:sh, :],           # верх
        gray[h-sh:, :],         # низ
        gray[:, :sw],           # ліво
        gray[:, w-sw:],         # право
    ]

    dark_strip_count = 0
    for strip in strips:
        if strip.size == 0:
            continue
        dark_ratio = float(np.mean(strip < SCREEN_EDGE_DARK_THRESHOLD))
        if dark_ratio > SCREEN_EDGE_DARK_RATIO:
            dark_strip_count += 1

    return dark_strip_count >= 2


def _detect_phone_camera(small: np.ndarray, bg_uniformity: float) -> bool:
    """
    Визначає, чи зроблено фото на камеру телефону за кольоровим відтінком фону.

    small: зменшене зображення BGR.
    bg_uniformity: background_uniformity метрика (0..1).
    """
    if bg_uniformity < 0.4:
        return False

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Пікселі світлого фону (L > 180)
    bg_mask = l_ch > 180
    bg_pixel_count = np.count_nonzero(bg_mask)
    if bg_pixel_count < 0.05 * small.shape[0] * small.shape[1]:
        return False

    b_bg = b_ch[bg_mask]
    b_median = float(np.median(b_bg))

    return b_median > PHONE_WARM_B_THRESHOLD or b_median < PHONE_COOL_B_THRESHOLD


def classify_capture_conditions(
    image: np.ndarray,
    background_uniformity: float = 0.5,
) -> str:
    """
    Визначає умови зйомки зображення.

    Повертає одне з:
      "screen_capture" — захоплення екрану
      "phone_camera"   — фото на телефон (кольоровий відтінок)
      "flat_uniform"   — рівномірний фон (скан/плоский об'єкт)
      "unknown"        — не визначено

    image: BGR uint8.
    background_uniformity: метрика однорідності фону (0..1).
    """
    # Ресайз до 300px по більшій стороні для швидкості
    h, w = image.shape[:2]
    max_side = max(h, w)
    scale = min(300.0 / max_side, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1. Перевірка на screen capture
    if _detect_screen_capture(small):
        return CAPTURE_SCREEN

    # 2. Рівномірний фон
    if background_uniformity > FLAT_UNIFORM_BG_THRESHOLD:
        if _detect_phone_camera(small, background_uniformity):
            return CAPTURE_PHONE
        return CAPTURE_FLAT

    return CAPTURE_UNKNOWN
