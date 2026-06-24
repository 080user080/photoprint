"""
Модуль для вимірювання та виправлення нахилу (skew) документа.
Використовується як попередній крок перед авто-перспективою.
"""

import numpy as np
import cv2

# Константи
DESKEW_MIN_ANGLE = 0.5          # менше цього — не повертати
DESKEW_MAX_ANGLE = 45.0         # більше цього — ігнорувати як шум
DESKEW_HOUGH_THRESHOLD = 50     # мінімальна кількість голосів для лінії
DESKEW_HOUGH_MIN_LENGTH = 100   # мінімальна довжина лінії в пікселях
DESKEW_HOUGH_MAX_GAP = 10       # максимальний розрив у лінії
DESKEW_RESIZE_MAX = 800         # розмір для аналізу (швидкість)
DESKEW_ANGLE_FILTER_LOW = -45.0 # нижня межа кута для фільтрації
DESKEW_ANGLE_FILTER_HIGH = 45.0 # верхня межа кута


def measure_skew_angle(image: np.ndarray) -> float:
    """
    Вимірює кут нахилу документа за допомогою HoughLinesP.

    1. Зменшує зображення до DESKEW_RESIZE_MAX по більшій стороні
    2. Конвертує в grayscale
    3. Adaptive threshold (THRESH_BINARY_INV) для виділення тексту
    4. HoughLinesP для пошуку ліній
    5. Обчислює кути кожної лінії
    6. Фільтрує кути в діапазоні -45..45 градусів
    7. Повертає медіану кутів або 0.0 якщо ліній менше 3

    Returns:
        Кут нахилу в градусах (додатний = проти годинникової стрілки).
    """
    h, w = image.shape[:2]
    scale = min(DESKEW_RESIZE_MAX / max(h, w), 1.0)
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold для виділення темного тексту на світлому фоні
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=DESKEW_HOUGH_THRESHOLD,
        minLineLength=DESKEW_HOUGH_MIN_LENGTH,
        maxLineGap=DESKEW_HOUGH_MAX_GAP,
    )

    if lines is None or len(lines) < 3:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        # Фільтруємо: залишаємо тільки кути в діапазоні
        if DESKEW_ANGLE_FILTER_LOW <= angle <= DESKEW_ANGLE_FILTER_HIGH:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0

    return float(np.median(angles))


def apply_deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Повертає зображення на заданий кут з білим фоном.

    Якщо |angle| < DESKEW_MIN_ANGLE — повертає копію без змін.
    """
    if abs(angle) < DESKEW_MIN_ANGLE:
        return image.copy()

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Обчислюємо новий розмір canvas, щоб зображення не обрізалось
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Коригуємо центр для нового розміру
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    result = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return result