"""
Модуль автоматичного білізнення фону для документів.
Визначає світлий однорідний фон і замінює його на білий (255, 255, 255),
зберігаючи контент (текст, лінії, малюнки).
"""

import cv2
import numpy as np

# Константи для визначення фону
WHITE_BG_L_THRESHOLD = 200.0    # пікселі з L > цього — потенційний фон
WHITE_BG_STD_THRESHOLD = 8.0    # локальний std нижче цього — однорідна ділянка
UNIFORMITY_MIN = 0.35           # мінімальна частка рівного фону для застосування
DETAIL_DENSITY_MAX = 0.35       # максимальна деталізація для застосування (вище — фото)
MORPH_KERNEL_SIZE = 7           # розмір ядра для морфологічних операцій
DILATE_KERNEL_SIZE = 5          # розмір ядра для розширення маски (згладжування країв)


def _local_std_map(gray: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Локальне стандартне відхилення через box filter (O(1) на піксель)."""
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(f * f, -1, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def make_background_white(
    image: np.ndarray,
    background_uniformity: float,
    detail_density: float,
) -> tuple[np.ndarray, bool]:
    """
    Замінює світлий однорідний фон на білий.

    Алгоритм:
    1. Переводить в LAB, бере L-канал.
    2. Будує маску фону: L > WHITE_BG_L_THRESHOLD + локальний std < WHITE_BG_STD_THRESHOLD.
    3. Морфологічне closing для усунення дрібних дірок.
    4. Розширення маски (dilate) для згладжування країв.
    5. Замінює пікселі фону на (255, 255, 255).

    Параметри:
        image: BGR зображення (uint8).
        background_uniformity: частка рівного фону (0..1) з діагностики.
        detail_density: щільність деталей (0..1) з діагностики.

    Повертає:
        (модифіковане_зображення, чи_була_заміна)
    """
    # Перевірка: чи варто застосовувати
    if background_uniformity < UNIFORMITY_MIN:
        return image.copy(), False
    if detail_density > DETAIL_DENSITY_MAX:
        return image.copy(), False

    result = image.copy()
    h, w = result.shape[:2]

    # LAB для L-каналу
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)

    # Маска 1: світлі пікселі
    light_mask = l_ch > WHITE_BG_L_THRESHOLD

    # Маска 2: однорідні ділянки
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    std_map = _local_std_map(gray)
    uniform_mask = std_map < WHITE_BG_STD_THRESHOLD

    # Комбінована маска: світлі + однорідні
    bg_mask = (light_mask & uniform_mask).astype(np.uint8) * 255

    # Морфологічне closing: усуває дрібні дірки в масці
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close)

    # Розширення маски для згладжування країв
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
    bg_mask = cv2.dilate(bg_mask, kernel_dilate)

    # Частка фону після морфології
    bg_ratio = float(np.mean(bg_mask > 0))

    # Якщо фон займає менше 10% — не застосовуємо (може бути помилковою маскою)
    if bg_ratio < 0.1:
        return image.copy(), False

    # Заміна фону на білий (255, 255, 255)
    result[bg_mask > 0] = [255, 255, 255]

    return result, True