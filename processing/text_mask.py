"""
Детектор текстових/однотонних областей на зображенні документа.
Використовується для адаптивного HDR — щоб не застосовувати HDR до тексту.

Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для детекції тексту
TEXT_MASK_BLOCK_SIZE = 15  # розмір блоку для adaptiveThreshold
TEXT_MASK_C = 5  # константа для adaptiveThreshold
TEXT_MASK_MORPH_KERNEL = (3, 3)  # ядро для морфологічного закриття дрібних прогалин
TEXT_MASK_DILATE_KERNEL = (7, 7)  # ядро для розширення текстової маски (захопити крайові пікселі)
TEXT_MASK_BLUR_SIGMA = 2.0  # sigma для blur перед бінаризацією (зменшення шуму)
TEXT_MASK_SMALL_COMPONENT_AREA = 30  # мінімальна площа зв'язної компоненти для тексту

# Константи для визначення "однотонності" області
UNIFORM_STD_THRESHOLD = 8.0  # std < цього = однотонна ділянка (фон документа)
UNIFORM_BLOCK_SIZE = 16  # розмір блоку для розбиття на ділянки


def _binarize_text(gray: np.ndarray) -> np.ndarray:
    """
    Адаптивна бінаризація для виділення тексту.
    Повертає бінарну маску (0/255), де 255 — текст (темні деталі на світлому фоні).
    """
    # Згладжуємо шум
    blurred = cv2.GaussianBlur(gray, (0, 0), TEXT_MASK_BLUR_SIGMA)
    
    # Адаптивна бінаризація — знаходить текст як темні області
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # INV — текст стає 255 (білим)
        TEXT_MASK_BLOCK_SIZE,
        TEXT_MASK_C,
    )
    return binary


def _filter_small_components(binary: np.ndarray) -> np.ndarray:
    """
    Видаляє дрібні шумові компоненти з бінарної маски.
    Залишає лише компоненти площею >= TEXT_MASK_SMALL_COMPONENT_AREA.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 2:
        return binary
    
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):  # 0 — фон
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= TEXT_MASK_SMALL_COMPONENT_AREA:
            mask[labels == i] = 255
    return mask


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """
    Розширює текстову маску, щоб захопити краї символів і ділянки
    з переходом між текстом і фоном.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TEXT_MASK_DILATE_KERNEL)
    return cv2.dilate(mask, kernel, iterations=2)


def text_region_mask(gray: np.ndarray) -> np.ndarray:
    """
    Повертає бінарну маску (uint8, 0/255) ділянок, схожих на текст:
    висока локальна щільність контрастних дрібних деталей.

    Маска = 255 там, де ймовірно текст (не застосовувати HDR).
    Маска = 0 там, де фон/однотонна область (можна застосовувати HDR).

    Алгоритм:
    1. Адаптивна бінаризація (INV — текст стає білим).
    2. Видалення дрібних шумових компонент.
    3. Морфологічне розширення для захоплення країв.
    """
    # 1. Бінаризація
    binary = _binarize_text(gray)
    
    # 2. Фільтрація дрібних шумів
    mask = _filter_small_components(binary)
    
    # 3. Розширення
    mask = _dilate_mask(mask)
    
    return mask


def uniform_area_mask(gray: np.ndarray) -> np.ndarray:
    """
    Повертає бінарну маску (uint8, 0/255) однотонних областей (фон документа).
    
    Маска = 255 там, де область однотонна (фон — HDR можна застосовувати).
    Маска = 0 там, де є текстура/деталі.

    Алгоритм:
    1. Розбиває зображення на блоки UNIFORM_BLOCK_SIZE x UNIFORM_BLOCK_SIZE.
    2. Для кожного блоку обчислює std.
    3. Якщо std < UNIFORM_STD_THRESHOLD — блок однотонний.
    """
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    block = UNIFORM_BLOCK_SIZE
    for y in range(0, h, block):
        y_end = min(y + block, h)
        for x in range(0, w, block):
            x_end = min(x + block, w)
            block_roi = gray[y:y_end, x:x_end]
            std = float(np.std(block_roi))
            if std < UNIFORM_STD_THRESHOLD:
                mask[y:y_end, x:x_end] = 255
    return mask