"""Класифікація типу зображення: bw_document / color_document / photo."""

from typing import Literal
import cv2
import numpy as np

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


def classify(
    image: np.ndarray,
    bw_std_thresh: float = 20.0,
    edge_ratio_min: float = 0.03,
    line_count_min: int = 3,
) -> DocType:
    """Повертає тип документа: 'bw_document' | 'color_document' | 'photo'."""
    small = cv2.resize(image, (0, 0), fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    std_a = float(np.std(a_ch))
    std_b = float(np.std(b_ch))

    edges = cv2.Canny(l_ch, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)

    min_line_length = min(small.shape[:2]) // HOUGH_MIN_LINE_LENGTH_RATIO
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
                            minLineLength=min_line_length, maxLineGap=HOUGH_MAX_LINE_GAP)
    line_count = len(lines) if lines is not None else 0

    # Чорно-білий
    if std_a < bw_std_thresh and std_b < bw_std_thresh:
        if edge_ratio >= edge_ratio_min and line_count >= line_count_min:
            return "bw_document"
        return "photo"

    # Кольоровий — документ чи фото
    if edge_ratio >= edge_ratio_min and line_count >= line_count_min:
        return "color_document"
    return "photo"
