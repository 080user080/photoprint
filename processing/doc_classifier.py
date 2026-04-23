"""Класифікація типу зображення: bw_document / color_document / photo."""

import cv2
import numpy as np


def classify(
    image: np.ndarray,
    bw_std_thresh: float = 20.0,
    edge_ratio_min: float = 0.03,
    line_count_min: int = 3,
) -> str:
    small = cv2.resize(image, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    std_a = float(np.std(a_ch))
    std_b = float(np.std(b_ch))

    edges = cv2.Canny(l_ch, 50, 150)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min(small.shape[:2]) // 5, maxLineGap=10)
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
