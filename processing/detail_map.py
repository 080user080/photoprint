"""
Карта локальной детализации изображения.
Используется для ограничения эффекта HDR/CLAHE на однотонных участках
(фон документа, небо, ровная бумага) — там, где CLAHE больше всего усиливает шум.

Не зависит от какого-либо другого модуля проекта.
"""

import cv2
import numpy as np

# Размер окна для вычисления локального std (нечётное число)
DETAIL_STD_KERNEL = 9

# Sigma для финального размытия карты (плавные переходы, без границ тайлов/блоков)
DETAIL_SMOOTH_SIGMA = 5.0

# Перцентиль локального std, принимаемый за "полную детализацию" (mask=1.0)
DETAIL_REF_PERCENTILE = 90.0

# Нижняя граница ref_std — защита от деления на ~0 на полностью плоских изображениях
DETAIL_MIN_REF_STD = 3.0


def local_std_map(gray: np.ndarray, kernel: int = DETAIL_STD_KERNEL) -> np.ndarray:
    """
    Локальное стандартное отклонение каждого пикселя через box filter.
    Быстро (O(n)), без поэлементных циклов.
    Возвращает float32 того же размера, что и gray.
    """
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(f * f, -1, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def detail_mask(gray: np.ndarray, noise_floor: float = None) -> np.ndarray:
    """
    Возвращает float32-карту 0..1:
      0.0 — однотонный участок (фон, бумага, шум) — эффект HDR должен быть ~0
      1.0 — участок с реальной текстурой/деталями — полный эффект HDR

    ref_std вычисляется из самого изображения (DETAIL_REF_PERCENTILE),
    поэтому один и тот же код одинаково корректно работает и на контрастных фото,
    и на почти плоских сканах — это и есть "универсальность" без слайдеров.

    noise_floor: если передан, заменяет DETAIL_MIN_REF_STD как нижнюю границу ref_std.
    Позволяет увеличить порог на шумных изображениях, чтобы не усиливать шум на фоне.
    Если None — вычисляется автоматически как медиана std на тихих 10% пикселей * 2.
    """
    std_map = local_std_map(gray)

    # Автоматическое вычисление noise_floor из изображения если не передан
    if noise_floor is None:
        noise_floor = max(float(np.percentile(std_map, 10)) * 2.0, DETAIL_MIN_REF_STD)

    ref_std = max(float(np.percentile(std_map, DETAIL_REF_PERCENTILE)), noise_floor)
    mask = np.clip(std_map / ref_std, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), DETAIL_SMOOTH_SIGMA)
    return mask
