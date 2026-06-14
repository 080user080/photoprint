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

# ---- Нові константи (PRIO 2) ----
PARTIAL_SKEW_THRESHOLD_RATIO = 0.01  # 1% від ширини/висоти документа

# ---- Нові константи (PRIO 3) ----

# 3.1: Border + solidity перевірка
BORDER_MARGIN_PX = 2
MAX_BORDER_AREA_RATIO = 0.92  # контур, що майже = всьому кадру — підозра на фон/стіл
MIN_SOLIDITY = 0.85

# 3.2: Адаптивний epsilon для approxPolyDP
APPROX_POLY_EPSILONS = (0.01, 0.02, 0.03, 0.04)

# 3.3: Sub-pixel уточнення кутів
SUBPIX_WIN_SIZE = (5, 5)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# 3.4: CLAHE перед бінаризацією
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# ---------------------------------------------------------------------------
# Публічний API
# ---------------------------------------------------------------------------

def auto_detect_corners(image: np.ndarray, max_dim: int = MAX_ANALYSIS_DIM) -> np.ndarray | None:
    """
    Автоматично знаходить 4 кути документа.

    Використовує множинні методи:
    1. Adaptive Threshold (для ч-б документів)
       + CLAHE-версія при нерівному освітленні (1b)
    2. Canny Edge Detection (для кольорових/фото)
       + CLAHE-версія (2b)
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
    Спроби в порядку надійності + CLAHE-версії при нерівному освітленні.
    """
    logger.debug("_detect_corners_impl: початок пошуку кутів")

    # Спроба 1: Adaptive Threshold (для ч-б документів)
    corners = _try_adaptive_threshold(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через Adaptive Threshold")
        corners = _refine_corners_subpix(gray, corners)
        return corners

    logger.debug("_detect_corners_impl: Adaptive Threshold не знайшов, пробуємо Canny")

    # Спроба 2: Canny Edge Detection (для кольорових/фото)
    corners = _try_canny(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через Canny")
        corners = _refine_corners_subpix(gray, corners)
        return corners

    logger.debug("_detect_corners_impl: Canny не знайшов, пробуємо CLAHE + Adaptive Threshold")

    # Спроба 1b: CLAHE + Adaptive Threshold (для нерівного освітлення)
    clahe_gray = _apply_clahe(gray)
    corners = _try_adaptive_threshold(clahe_gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через CLAHE + Adaptive Threshold")
        corners = _refine_corners_subpix(gray, corners)
        return corners

    logger.debug("_detect_corners_impl: CLAHE + Adaptive Threshold не знайшов, пробуємо CLAHE + Canny")

    # Спроба 2b: CLAHE + Canny (для нерівного освітлення)
    corners = _try_canny(clahe_gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через CLAHE + Canny")
        corners = _refine_corners_subpix(gray, corners)
        return corners

    logger.debug("_detect_corners_impl: CLAHE+Canny не знайшов, пробуємо fallback bounding box")

    # Спроба 3: Fallback — найбільший bounding box
    corners = _try_largest_contour(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через fallback bounding box")
        corners = _refine_corners_subpix(gray, corners)
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

        corners = _find_quad_contour(closed, gray.shape)
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

    return _find_quad_contour(closed, gray.shape)


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

    # Сортуємо за площею, фільтруємо border-контури
    h, w = gray.shape
    image_area = h * w
    candidates = [c for c in contours
                  if not _touches_image_border(c, gray.shape)
                  and cv2.contourArea(c) < image_area * MAX_BORDER_AREA_RATIO]
    if not candidates:
        candidates = contours  # якщо всі відфільтрувались — використовуємо всі

    largest = max(candidates, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Валідація пропорцій
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect > MAX_ASPECT_RATIO or aspect < 1.2:  # Документ має бути "прямокутним"
        return None

    # Валідація площі
    box_area = bw * bh
    if box_area < image_area * MIN_DOCUMENT_AREA_RATIO:
        return None

    # Повертаємо 4 кути bounding box
    corners = np.array([
        [x, y],          # TL
        [x + bw, y],     # TR
        [x + bw, y + bh],  # BR
        [x, y + bh]      # BL
    ], dtype=np.float32)

    return corners


def _find_quad_contour(binary: np.ndarray, gray_shape: tuple) -> np.ndarray | None:
    """
    Шукає 4-кутний контур-документ з валідацією.
    Тепер використовує адаптивний APPROX_POLY_EPSILONS та solidity-перевірку.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Сортуємо за площею
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = gray_shape
    image_area = h * w

    for cnt in contours[:CONTOURS_TO_CHECK]:
        # Фільтр площі
        area = cv2.contourArea(cnt)
        if area < image_area * MIN_DOCUMENT_AREA_RATIO:
            continue

        # Solidify-перевірка (компактність форми)
        solidity_val = _solidity(cnt)
        if solidity_val < MIN_SOLIDITY:
            continue

        # Border-перевірка (контур не має торкатись межі кадру, якщо площа велика)
        if _touches_image_border(cnt, gray_shape) and area > image_area * MAX_BORDER_AREA_RATIO:
            continue

        # Адаптивний epsilon: пробуємо кілька значень
        peri = cv2.arcLength(cnt, True)
        for eps in APPROX_POLY_EPSILONS:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == CORNER_COUNT:
                if _validate_document(approx, (h, w)):
                    return approx.reshape(4, 2).astype(np.float32)

        # Якщо більше 4 кутів — беремо bounding box як fallback
        # (використовуємо останній approx з найбільшим epsilon)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILONS[-1] * peri, True)
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


# ---------------------------------------------------------------------------
# PRIO 2 — Часткова корекція перспективи
# ---------------------------------------------------------------------------

def detect_skewed_sides(pts: np.ndarray) -> dict[str, bool]:
    """
    Визначає, які сторони документа є "кривими".

    pts: впорядковані [TL, TR, BR, BL].
    Повертає {"top": bool, "bottom": bool, "left": bool, "right": bool} —
    True, якщо відповідна пара кутів відхиляється від "прямої" сторони
    більше ніж на PARTIAL_SKEW_THRESHOLD_RATIO від розміру документа.
    """
    tl, tr, br, bl = pts
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl), 1.0)
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1.0)
    thr_w = width * PARTIAL_SKEW_THRESHOLD_RATIO
    thr_h = height * PARTIAL_SKEW_THRESHOLD_RATIO
    return {
        "top":    abs(tl[1] - tr[1]) > thr_h,
        "bottom": abs(bl[1] - br[1]) > thr_h,
        "left":   abs(tl[0] - bl[0]) > thr_w,
        "right":  abs(tr[0] - br[0]) > thr_w,
    }


def apply_partial_correction(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Виправляє перспективу лише для "кривих" сторін документа.
    Якщо жодна сторона не крива — повертає копію без змін.
    Якщо всі 4 сторони криві — делегує в apply_correction (повна корекція).

    Args:
        image: BGR зображення
        corners: float32 array shape (4,2), будь-який порядок

    Returns:
        Випрямлене BGR зображення або копія без змін
    """
    pts = _order_points(corners.astype(np.float32))
    skew = detect_skewed_sides(pts)

    if not any(skew.values()):
        return image.copy()
    if all(skew.values()):
        return apply_correction(image, corners)

    adjusted = pts.copy()
    tl, tr, br, bl = 0, 1, 2, 3

    if skew["top"]:
        y = (pts[tl][1] + pts[tr][1]) / 2.0
        adjusted[tl][1] = y
        adjusted[tr][1] = y
    if skew["bottom"]:
        y = (pts[bl][1] + pts[br][1]) / 2.0
        adjusted[bl][1] = y
        adjusted[br][1] = y
    if skew["left"]:
        x = (pts[tl][0] + pts[bl][0]) / 2.0
        adjusted[tl][0] = x
        adjusted[bl][0] = x
    if skew["right"]:
        x = (pts[tr][0] + pts[br][0]) / 2.0
        adjusted[tr][0] = x
        adjusted[br][0] = x

    # dst — той самий bounding rect, що й adjusted-точки
    dst, width, height = _compute_destination(adjusted)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        image, M, (width, height),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def auto_correct_partial(image: np.ndarray, max_dim: int = MAX_ANALYSIS_DIM) -> tuple[np.ndarray, bool]:
    """
    Авто-детект + часткова корекція перспективи в одному виклику.
    Якщо документ знайдено — застосовує apply_partial_correction.
    Інакше — повертає копію оригіналу.

    Returns:
        (результат, знайдено)
    """
    corners = auto_detect_corners(image, max_dim=max_dim)
    if corners is None:
        return image.copy(), False
    return apply_partial_correction(image, corners), True


# ---------------------------------------------------------------------------
# Нові функції (PRIO 3)
# ---------------------------------------------------------------------------

def _touches_image_border(cnt: np.ndarray, shape: tuple, margin: int = BORDER_MARGIN_PX) -> bool:
    """
    Перевіряє, чи торкається контур межі зображення.

    Args:
        cnt: контур (numpy array)
        shape: (height, width) зображення
        margin: допустимий відступ від краю в пікселях

    Returns:
        True, якщо контур торкається або майже торкається краю зображення
    """
    h, w = shape[:2]
    x, y, bw, bh = cv2.boundingRect(cnt)
    return (x <= margin or y <= margin
            or (x + bw) >= (w - margin)
            or (y + bh) >= (h - margin))


def _solidity(cnt: np.ndarray) -> float:
    """
    Обчислює компактність форми (solidity = area / convex_hull_area).

    Для документа очікується solidity >= 0.85 (форма близька до опуклої).
    Для "рамки фото / столу" solidity буде низькою через нерівні краї.

    Returns:
        Solidity (0.0 — 1.0). Якщо cnt має < 3 точок або hull_area == 0, повертає 0.0.
    """
    if len(cnt) < 3:
        return 0.0
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return area / hull_area


def _refine_corners_subpix(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Sub-pixel уточнення позицій кутів через cornerSubPix.

    Args:
        gray: grayscale зображення (оригінальний ~800px)
        corners: float32 array shape (4,2)

    Returns:
        Уточнені кути (або оригінальні, якщо cornerSubPix кинув помилку)
    """
    try:
        pts = corners.reshape(-1, 1, 2).astype(np.float32)
        refined = cv2.cornerSubPix(gray, pts, SUBPIX_WIN_SIZE, SUBPIX_ZERO_ZONE, SUBPIX_CRITERIA)
        return refined.reshape(-1, 2)
    except Exception:
        logger.debug("_refine_corners_subpix: cornerSubPix кинув помилку, повертаємо оригінальні кути")
        return corners


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Застосовує CLAHE (Contrast Limited Adaptive Histogram Equalization)
    до grayscale зображення для вирівнювання нерівномірного освітлення.

    Returns:
        Оброблений grayscale масив тієї ж форми
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)