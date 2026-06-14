"""Класифікація типу зображення: bw_document / color_document / photo."""

from typing import Literal
import cv2
import numpy as np
import logging

# Типи документів для type hints
DocType = Literal["bw_document", "color_document", "photo"]

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


def classify(
    image: np.ndarray,
    bw_std_thresh: float = 10.0,
    edge_ratio_min: float = 0.03,
    line_count_min: int = 3,
) -> DocType:
    """
    Повертає тип документа: 'bw_document' | 'color_document' | 'photo'.
    
    bw_std_thresh: поріг стандартного відхилення a/b каналів у LAB.
        Зменшено з 20.0 до 10.0 для запобігання помилковій класифікації
        кольорових зображень з низькою насиченістю як чорно-білих документів.
    """
    small = cv2.resize(image, (0, 0), fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

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
