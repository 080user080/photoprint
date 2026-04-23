"""Класифікація типу зображення: bw_document / color_document / photo."""

import cv2
import numpy as np

STD_AB_BW_THRESH = 20.0
EDGE_RATIO_DOC_MIN = 0.03
LINE_COUNT_DOC_MIN = 3


def classify(image: np.ndarray) -> str:
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
    if std_a < STD_AB_BW_THRESH and std_b < STD_AB_BW_THRESH:
        if edge_ratio >= EDGE_RATIO_DOC_MIN and line_count >= LINE_COUNT_DOC_MIN:
            return "bw_document"
        return "photo"

    # Кольоровий — документ чи фото
    if edge_ratio >= EDGE_RATIO_DOC_MIN and line_count >= LINE_COUNT_DOC_MIN:
        return "color_document"
    return "photo"
