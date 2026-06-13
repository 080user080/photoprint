"""
Перспективна корекція документа.
Авто: знаходить 4 кути документа через множинні методи детекції.
Ручна: приймає 4 точки від користувача.
Не залежить від GUI модулів.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

# Розмір для аналізу (швидкість + стабільність)
MAX_ANALYSIS_DIM = 800

# Gaussian blur
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0

# Morphology
MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS = 1

# Canny edges
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150

# Adaptive threshold
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Контури
CONTOURS_TO_CHECK = 5
MIN_DOCUMENT_AREA_RATIO = 0.05  # мінімум 5% площі
MAX_ASPECT_RATIO = 5.0  # максимальне співвідношення сторін (не 1:50)
APPROX_POLY_EPSILON = 0.02

# Кількість кутів документа
CORNER_COUNT = 4

# Padding для уникнення обрізання по краях
PADDING_RATIO = 0.06  # 6% від розміру з кожного боку


# ---------------------------------------------------------------------------
# Публічний API
# ---------------------------------------------------------------------------

def auto_detect_corners(image: np.ndarray, max_dim: int = MAX_ANALYSIS_DIM) -> np.ndarray | None:
    """
    Автоматично знаходить 4 кути документа.
    
    Використовує множинні методи:
    1. Adaptive Threshold (для ч-б документів)
    2. Canny Edge Detection (для кольорових/фото)
    3. Fallback — bounding box найбільшого контуру
    
    Args:
        image: BGR зображення (будь-якого розміру)
        max_dim: максимальний розмір для аналізу (швидкість)
    
    Returns:
        Масив float32 shape (4,2) у порядку [TL, TR, BR, BL] або None
    """
    h, w = image.shape[:2]
    logger.debug(f"auto_detect_corners: вхідний розмір {w}x{h}")
    scale = 1.0
    
    # Resize для швидкості та меншого шуму
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
        logger.debug(f"auto_detect_corners: resize до {small.shape[1]}x{small.shape[0]}, scale={scale:.3f}")
    else:
        small = image
        logger.debug(f"auto_detect_corners: resize не потрібен")
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    corners_small = _detect_corners_impl(gray)
    
    if corners_small is None:
        logger.debug("auto_detect_corners: кути не знайдено")
        return None
    
    # Масштабуємо назад
    corners = corners_small / scale
    logger.debug(f"auto_detect_corners: кути знайдено, масштабовано назад: {corners}")
    return corners.astype(np.float32)


def apply_correction(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Виконує перспективну трансформацію за 4 кутами.
    
    Args:
        image: BGR зображення
        corners: float32 array shape (4,2), будь-який порядок
    
    Returns:
        Випрямлений BGR uint8
    """
    logger.debug(f"apply_correction: вхідний розмір {image.shape[1]}x{image.shape[0]}")
    logger.debug(f"apply_correction: кути {corners}")
    pts = _order_points(corners.astype(np.float32))
    dst, width, height = _compute_destination(pts)
    logger.debug(f"apply_correction: вихідний розмір {width}x{height}")
    M = cv2.getPerspectiveTransform(pts, dst)
    # Білий колір для padding замість чорного
    warped = cv2.warpPerspective(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    logger.debug(f"apply_correction: результат {warped.shape[1]}x{warped.shape[0]}")
    return warped


def auto_correct(image: np.ndarray, max_dim: int = MAX_ANALYSIS_DIM) -> tuple[np.ndarray, bool]:
    """
    Зручна обгортка: авто-детект + корекція в одному виклику.
    
    Returns:
        (результат, True) якщо документ знайдено,
        (оригінал, False) якщо ні
    """
    corners = auto_detect_corners(image, max_dim=max_dim)
    if corners is None:
        return image.copy(), False
    return apply_correction(image, corners), True


# ---------------------------------------------------------------------------
# Внутрішні функції
# ---------------------------------------------------------------------------

def _detect_corners_impl(gray: np.ndarray) -> np.ndarray | None:
    """
    Внутрішня функція — шукає кути на зображенні ~800px.
    Спроби в порядку надійності.
    """
    logger.debug("_detect_corners_impl: початок пошуку кутів")
    
    # Спроба 1: Adaptive Threshold (для ч-б документів)
    corners = _try_adaptive_threshold(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через Adaptive Threshold")
        return corners
    
    logger.debug("_detect_corners_impl: Adaptive Threshold не знайшов, пробуємо Canny")
    
    # Спроба 2: Canny Edge Detection (для кольорових/фото)
    corners = _try_canny(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через Canny")
        return corners
    
    logger.debug("_detect_corners_impl: Canny не знайшов, пробуємо fallback bounding box")
    
    # Спроба 3: Fallback — найбільший bounding box
    corners = _try_largest_contour(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через fallback bounding box")
        return corners
    
    logger.debug("_detect_corners_impl: жоден метод не знайшов кути")
    return None


def _try_adaptive_threshold(gray: np.ndarray) -> np.ndarray | None:
    """
    Перша спроба — через adaptive threshold.
    Пробуємо обидва варіанти: THRESH_BINARY та THRESH_BINARY_INV
    (документ може бути світлішим або темнішим за фон)
    """
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    
    for thresh_method in [cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_method, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
        
        # Морфологія для закриття прогалин
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
        
        corners = _find_quad_contour(closed)
        if corners is not None:
            return corners
    
    return None


def _try_canny(gray: np.ndarray) -> np.ndarray | None:
    """
    Друга спроба — через Canny edge detection.
    Кращий для кольорових документів з різними фонами.
    """
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    
    # Морфологія для з'єднання країв
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    
    return _find_quad_contour(closed)


def _try_largest_contour(gray: np.ndarray) -> np.ndarray | None:
    """
    Третя спроба (fallback) — беремо найбільший контур та його bounding box.
    Працює навіть коли документ не ідеальний прямокутник.
    """
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Валідація пропорцій
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO or aspect < 1.2:  # Документ має бути "прямокутним"
        return None
    
    # Валідація площі
    image_area = gray.shape[0] * gray.shape[1]
    box_area = w * h
    if box_area < image_area * MIN_DOCUMENT_AREA_RATIO:
        return None
    
    # Повертаємо 4 кути bounding box
    corners = np.array([
        [x, y],      # TL
        [x + w, y],  # TR
        [x + w, y + h],  # BR
        [x, y + h]   # BL
    ], dtype=np.float32)
    
    return corners


def _find_quad_contour(binary: np.ndarray) -> np.ndarray | None:
    """
    Шукає 4-кутний контур-документ з валідацією.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Сортуємо за площею
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = binary.shape
    image_area = h * w
    
    for cnt in contours[:CONTOURS_TO_CHECK]:
        # Фільтр площі
        area = cv2.contourArea(cnt)
        if area < image_area * MIN_DOCUMENT_AREA_RATIO:
            continue
        
        # Апроксимація до полігону
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILON * peri, True)
        
        # Шукаємо 4 кути
        if len(approx) == CORNER_COUNT:
            if _validate_document(approx, (h, w)):
                return approx.reshape(4, 2).astype(np.float32)
        
        # Якщо більше 4 кутів — беремо bounding box як fallback
        if len(approx) > CORNER_COUNT:
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if 1.2 <= aspect <= MAX_ASPECT_RATIO:
                if (bw * bh) >= image_area * MIN_DOCUMENT_AREA_RATIO:
                    return np.array([
                        [x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]
                    ], dtype=np.float32)
    
    return None


def _validate_document(approx: np.ndarray, image_shape: tuple) -> bool:
    """
    Перевіряє чи це схоже на документ:
    1. Площа достатня
    2. Пропорції адекватні (не 1:50)
    """
    h, w = image_shape[:2]
    image_area = h * w
    
    # Площа
    doc_area = cv2.contourArea(approx)
    if doc_area < image_area * MIN_DOCUMENT_AREA_RATIO:
        return False
    
    # Пропорції bounding box
    x, y, bw, bh = cv2.boundingRect(approx)
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect > MAX_ASPECT_RATIO:
        return False
    
    return True


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядковує 4 точки: [TL, TR, BR, BL].
    """
    rect = np.zeros((CORNER_COUNT, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL — мінімальна сума x+y
    rect[2] = pts[np.argmax(s)]   # BR — максимальна сума x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR — мінімальна різниця y-x
    rect[3] = pts[np.argmax(diff)]  # BL — максимальна різниця y-x
    return rect


def _compute_destination(pts: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Обчислює розміри вихідного зображення та dst-точки.
    Додає padding щоб уникнути обрізання по краях.
    """
    tl, tr, br, bl = pts
    
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))
    
    # Додаємо padding щоб уникнути обрізання по краях
    padding_x = int(width * PADDING_RATIO)
    padding_y = int(height * PADDING_RATIO)
    width += 2 * padding_x
    height += 2 * padding_y
    
    dst = np.array([
        [padding_x, padding_y],
        [width - padding_x - 1, padding_y],
        [width - padding_x - 1, height - padding_y - 1],
        [padding_x, height - padding_y - 1],
    ], dtype=np.float32)
    
    return dst, width, height
