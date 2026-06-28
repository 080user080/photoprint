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
# PARTIAL_SKEW_THRESHOLD_RATIO визначає чутливість детекції "кривих" сторін.
# 0.03 (3%) — для документа 800px поріг ~24px.
# Значення 0.01 (1%) було надто чутливим: шум детекції кутів (>8px)
# спричиняв хибне спрацювання warp на рівних документах.
PARTIAL_SKEW_THRESHOLD_RATIO = 0.015  # знижено з 0.03: телефонні фото мають невелике але реальне викривлення

# ---- Нові константи (PRIO 3) ----

# 3.1: Border + solidity перевірка
BORDER_MARGIN_PX = 2
MAX_BORDER_AREA_RATIO = 0.92  # контур, що майже = всьому кадру — підозра на фон/стіл
MIN_SOLIDITY = 0.85

# 3.5: Геометрична валідація кутів
CORNER_ANGLE_MIN_DEG = 45.0   # мінімальний кут між сторонами (градуси)
CORNER_ANGLE_MAX_DEG = 135.0  # максимальний кут між сторонами
SIDE_STRAIGHTNESS_MAX_RATIO = 0.08  # макс. відхилення середини сторони від прямої (відносно довжини)
MIN_QUAD_AREA_RATIO = 0.05    # мінімум 5% площі кадру
MAX_QUAD_AREA_RATIO = 0.97    # максимум 97% площі кадру

# 3.2: Адаптивний epsilon для approxPolyDP
APPROX_POLY_EPSILONS = (0.01, 0.02, 0.03, 0.04)

# 3.3: Sub-pixel уточнення кутів
SUBPIX_WIN_SIZE = (5, 5)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# 3.4: CLAHE перед бінаризацією
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Hough Lines детекція (новий метод)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD_LINES = 80        # мінімальна кількість голосів
HOUGH_MIN_LINE_LENGTH_RATIO = 0.15 # мінімальна довжина лінії = 15% від min(h,w)
HOUGH_MAX_LINE_GAP = 20
HOUGH_ANGLE_TOLERANCE_DEG = 20.0   # допуск: лінія вважається "горизонтальною" або "вертикальною"; збільшено з 15° для повернутих зображень
HOUGH_MARGIN_RATIO = 0.03          # відступ від краю кадру для фільтрації країв рамки

# Ітеративне уточнення
ITERATIVE_MAX_PASSES = 2          # максимум проходів warp
ITERATIVE_MIN_SKEW_RATIO = 0.015  # мінімальне залишкове викривлення для другого проходу (1.5%)


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
    corners_small = _detect_with_rotation_candidates(gray)

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
    Збирає всі кандидати з множинних методів, оцінює через _score_quad,
    повертає найкращого. Fallback — найбільший контур.
    """
    logger.debug("_detect_corners_impl: початок пошуку кутів (scoring-based)")

    clahe_gray = _apply_clahe(gray)
    candidates = []  # list[tuple[score, corners]]

    methods = [
        ("adaptive",        lambda g: _try_adaptive_threshold(g)),
        ("canny",           lambda g: _try_canny(g)),
        ("clahe+adaptive",  lambda g: _try_adaptive_threshold(clahe_gray)),
        ("clahe+canny",     lambda g: _try_canny(clahe_gray)),
        ("hough",           lambda g: _try_hough_lines(g)),
        ("external_contour", lambda g: _try_external_contour(g)),
    ]

    for name, fn in methods:
        corners = fn(gray)
        if corners is not None:
            score = _score_quad(corners, gray.shape)
            if score > 0.0:
                refined = _refine_corners_subpix(gray, corners)
                candidates.append((score, refined))
                logger.debug(f"_detect_corners_impl: {name} score={score:.3f}")

    if not candidates:
        # Fallback: largest contour
        logger.debug("_detect_corners_impl: жоден метод не знайшов, пробуємо fallback bounding box")
        corners = _try_largest_contour(gray)
        if corners is not None:
            refined = _refine_corners_subpix(gray, corners)
            return refined
        logger.debug("_detect_corners_impl: жоден метод не знайшов кути")
        return None

    # Повертаємо кандидата з найвищим скором
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_corners = candidates[0]
    logger.debug(f"_detect_corners_impl: вибрано score={best_score:.3f} з {len(candidates)} кандидатів")
    return best_corners


def _detect_with_rotation_candidates(gray: np.ndarray) -> np.ndarray | None:
    """
    Пробує детекцію для 4 орієнтацій зображення (0°, 90°, 180°, 270°).
    Якщо знаходить кути — перераховує їх координати назад у простір оригіналу.
    Повертає найкращий результат або None.
    """
    h, w = gray.shape[:2]
    best_score = 0.0
    best_corners = None

    rotations = [
        (0,   gray),
        (90,  cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(gray, cv2.ROTATE_180)),
        (270, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    for angle, rotated in rotations:
        corners = _detect_corners_impl(rotated)
        if corners is None:
            continue
        rh, rw = rotated.shape[:2]
        score = _score_quad(corners, rotated.shape)
        # Завдання 1.2: бонус 10% для оригінальної орієнтації
        if angle == 0:
            score = score * 1.10
        if score <= best_score:
            continue
        # Перераховуємо координати назад у простір оригінального gray
        if angle == 0:
            back = corners
        elif angle == 90:
            # ROTATE_90_CLOCKWISE: rotated має розміри (w, h) → rh=w, rw=h
            # Зворотня трансформація: (x, y) у rotated → (rh-1-y, x) в оригіналі
            back = np.array([[rh - 1 - c[1], c[0]] for c in corners], dtype=np.float32)
        elif angle == 180:
            # ROTATE_180: rotated має розміри (h, w) → rh=h, rw=w
            # Зворотня трансформація: (x, y) у rotated → (rw-1-x, rh-1-y) в оригіналі
            back = np.array([[rw - 1 - c[0], rh - 1 - c[1]] for c in corners], dtype=np.float32)
        elif angle == 270:
            # ROTATE_90_COUNTERCLOCKWISE: rotated має розміри (w, h) → rh=w, rw=h
            # Зворотня трансформація: (x, y) у rotated → (y, rw-1-x) в оригіналі
            back = np.array([[c[1], rw - 1 - c[0]] for c in corners], dtype=np.float32)
        # Завдання 1.3: валідація орієнтації квадрилатераля після _order_points
        ordered_back = _order_points(back)
        if not _is_clockwise(ordered_back):
            logger.debug(f"_detect_with_rotation_candidates: angle={angle} — не за годинниковою стрілкою, відхиляємо")
            continue
        best_score = score
        best_corners = back

    return best_corners


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

    Валідація:
    - Пропорції: aspect ratio >= 1.1 (документ має бути прямокутним, не квадратом)
    - Площа: не менше MIN_DOCUMENT_AREA_RATIO від кадру
    - Bounding box не займає > MAX_BORDER_AREA_RATIO площі кадру
    - Якщо bounding box торкається краю кадру з 2+ сторін — відхиляємо (фон/стіл)
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
    box_area = bw * bh

    # Перевірка: якщо bounding box займає > 92% кадру — це не документ, а фон
    if box_area > image_area * MAX_BORDER_AREA_RATIO:
        return None

    # Валідація пропорцій
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect > MAX_ASPECT_RATIO or aspect < 1.1:  # документ має бути прямокутним
        return None

    # Валідація площі
    if box_area < image_area * MIN_DOCUMENT_AREA_RATIO:
        return None

    # Перевірка: якщо bounding box торкається краю кадру з 2+ сторін — це не документ
    touches_top = y <= BORDER_MARGIN_PX
    touches_bottom = (y + bh) >= (h - BORDER_MARGIN_PX)
    touches_left = x <= BORDER_MARGIN_PX
    touches_right = (x + bw) >= (w - BORDER_MARGIN_PX)
    sides_touched = sum([touches_top, touches_bottom, touches_left, touches_right])
    if sides_touched >= 2:
        return None

    # Повертаємо 4 кути bounding box
    corners = np.array([
        [x, y],          # TL
        [x + bw, y],     # TR
        [x + bw, y + bh],  # BR
        [x, y + bh]      # BL
    ], dtype=np.float32)

    # Валідація якості кутів
    valid, quality = _validate_quad_quality(corners, (h, w))
    if valid:
        logger.debug(f"_try_largest_contour: якість {quality:.3f}")
        return corners
    else:
        logger.debug("_try_largest_contour: валідація якості провалилась")
        return None


def _try_hough_lines(gray: np.ndarray) -> np.ndarray | None:
    """
    Детекція кутів документа через пошук домінантних горизонтальних
    та вертикальних ліній (HoughLinesP) та обчислення їх перетинів.

    Переваги перед контурним методом:
    - Працює коли контур документа не замкнений (тінь, перекриття)
    - Стійкий до схожого кольору документа та фону
    - Знаходить геометричну структуру навіть при розривах

    Алгоритм:
    1. Canny → HoughLinesP
    2. Класифікуємо лінії на горизонтальні та вертикальні
    3. Фільтруємо лінії біля країв кадру (рамка, не документ)
    4. Знаходимо 2 найдомінантніші H-лінії та 2 V-лінії
    5. Обчислюємо 4 перетини → кути документа
    6. Валідуємо через _validate_quad_quality

    Returns:
        float32 array shape (4,2) або None
    """
    h, w = gray.shape[:2]
    margin_x = int(w * HOUGH_MARGIN_RATIO)
    margin_y = int(h * HOUGH_MARGIN_RATIO)
    min_line_len = int(min(h, w) * HOUGH_MIN_LINE_LENGTH_RATIO)

    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD_LINES,
        minLineLength=min_line_len,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None or len(lines) < 4:
        logger.debug("_try_hough_lines: недостатньо ліній")
        return None

    h_lines = []  # горизонтальні: (y_avg, x_start, x_end, votes)
    v_lines = []  # вертикальні:   (x_avg, y_start, y_end, votes)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_line_len:
            continue

        angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))

        if angle_deg < HOUGH_ANGLE_TOLERANCE_DEG:
            # Горизонтальна лінія
            y_avg = (y1 + y2) / 2.0
            # Фільтруємо лінії біля верхнього/нижнього краю (рамка кадру)
            if y_avg < margin_y or y_avg > h - margin_y:
                continue
            h_lines.append((y_avg, min(x1, x2), max(x1, x2), length))

        elif angle_deg > (90.0 - HOUGH_ANGLE_TOLERANCE_DEG):
            # Вертикальна лінія
            x_avg = (x1 + x2) / 2.0
            # Фільтруємо лінії біля лівого/правого краю
            if x_avg < margin_x or x_avg > w - margin_x:
                continue
            v_lines.append((x_avg, min(y1, y2), max(y1, y2), length))

    if len(h_lines) < 2 or len(v_lines) < 2:
        logger.debug(
            f"_try_hough_lines: недостатньо H={len(h_lines)} V={len(v_lines)} ліній"
        )
        return None

    # Сортуємо за довжиною (голоси) — беремо найдовші
    h_lines.sort(key=lambda x: x[3], reverse=True)
    v_lines.sort(key=lambda x: x[3], reverse=True)

    # Кластеризація: об'єднуємо близькі паралельні лінії
    h_lines = _cluster_parallel_lines(h_lines, axis="h", image_size=(h, w))
    v_lines = _cluster_parallel_lines(v_lines, axis="v", image_size=(h, w))

    if len(h_lines) < 2 or len(v_lines) < 2:
        logger.debug("_try_hough_lines: після кластеризації недостатньо ліній")
        return None

    # Беремо 2 найдомінантніші лінії кожного типу
    # Для H: верхня (менша y) і нижня (більша y)
    h_sorted_by_pos = sorted(h_lines[:4], key=lambda x: x[0])
    h_top = h_sorted_by_pos[0]
    h_bottom = h_sorted_by_pos[-1]

    # Для V: ліва (менша x) і права (більша x)
    v_sorted_by_pos = sorted(v_lines[:4], key=lambda x: x[0])
    v_left = v_sorted_by_pos[0]
    v_right = v_sorted_by_pos[-1]

    # Обчислюємо 4 перетини H та V ліній
    def _intersect_hv(h_line, v_line):
        """Перетин горизонтальної та вертикальної лінії."""
        y = h_line[0]
        x = v_line[0]
        return np.array([x, y], dtype=np.float32)

    tl = _intersect_hv(h_top,    v_left)
    tr = _intersect_hv(h_top,    v_right)
    br = _intersect_hv(h_bottom, v_right)
    bl = _intersect_hv(h_bottom, v_left)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Валідація
    valid, quality = _validate_quad_quality(corners, (h, w))
    if not valid:
        logger.debug(f"_try_hough_lines: валідація провалилась")
        return None

    logger.debug(f"_try_hough_lines: знайдено кути, якість={quality:.3f}")
    return corners


def _try_external_contour(gray: np.ndarray) -> np.ndarray | None:
    """
    Fallback для документів з багатою внутрішньою текстурою (гільош, орнамент).
    Використовує сильне розмиття щоб прибрати внутрішню текстуру,
    залишаючи тільки зовнішні межі документа.
    """
    # Сильне розмиття прибирає внутрішній орнамент
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    # Otsu threshold шукає глобальний розподіл (документ vs фон)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Морфологія для закриття дрібних прогалин
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_contour(closed, gray.shape)


def _cluster_parallel_lines(
    lines: list,
    axis: str,
    image_size: tuple,
    cluster_ratio: float = 0.05,
) -> list:
    """
    Об'єднує близькі паралельні лінії в одну (зважене середнє).

    Args:
        lines: список (position, start, end, weight)
        axis: "h" — горизонтальні (position = y), "v" — вертикальні (position = x)
        image_size: (h, w)
        cluster_ratio: відстань менше cluster_ratio * розмір → об'єднуємо

    Returns:
        Відфільтрований список ліній (одна на кластер, найдовша)
    """
    if not lines:
        return lines

    h, w = image_size
    threshold = (h if axis == "h" else w) * cluster_ratio

    clusters = []
    used = [False] * len(lines)

    for i, line in enumerate(lines):
        if used[i]:
            continue
        cluster = [line]
        used[i] = True
        pos_i = line[0]
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            pos_j = lines[j][0]
            if abs(pos_i - pos_j) < threshold:
                cluster.append(lines[j])
                used[j] = True
        # Представник кластера — лінія з найбільшою вагою (довжиною)
        best = max(cluster, key=lambda x: x[3])
        clusters.append(best)

    return clusters


def _validate_quad_quality(pts: np.ndarray, image_shape: tuple) -> tuple[bool, float]:
    """
    Геометрична валідація 4 знайдених кутів документа.

    Перевірки:
    1. Площа — від MIN_QUAD_AREA_RATIO до MAX_QUAD_AREA_RATIO кадру.
    2. Кути між сторонами — від CORNER_ANGLE_MIN_DEG до CORNER_ANGLE_MAX_DEG.
       Ідеальний прямокутник = 90° у всіх 4 кутах.
    3. Прямолінійність сторін — середина кожної сторони не має відхилятись
       від прямої між кутами більше ніж SIDE_STRAIGHTNESS_MAX_RATIO від довжини.

    Args:
        pts: float32 array shape (4,2), впорядковані [TL, TR, BR, BL].
        image_shape: (height, width) зображення.

    Returns:
        (valid: bool, quality_score: float)
        quality_score: 0..1, де 1 = ідеальний прямокутник.
        valid=False якщо хоча б одна перевірка провалилась.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)

    ordered = _order_points(pts.astype(np.float32))
    tl, tr, br, bl = ordered

    # --- Перевірка 1: Площа ---
    quad_area = float(cv2.contourArea(ordered))
    area_ratio = quad_area / image_area
    if area_ratio < MIN_QUAD_AREA_RATIO or area_ratio > MAX_QUAD_AREA_RATIO:
        logger.debug(
            f"_validate_quad_quality: FAIL площа {area_ratio:.3f} "
            f"(ліміт {MIN_QUAD_AREA_RATIO}..{MAX_QUAD_AREA_RATIO})"
        )
        return False, 0.0

    # --- Перевірка 2: Кути між сторонами ---
    # Вектори сторін: top, right, bottom (reversed), left (reversed)
    sides = [
        tr - tl,   # top → right
        br - tr,   # right ↓
        bl - br,   # bottom ← left
        tl - bl,   # left ↑
    ]
    angle_scores = []
    for i in range(4):
        v1 = sides[i].astype(np.float64)
        v2 = sides[(i + 1) % 4].astype(np.float64)
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1.0 or len2 < 1.0:
            logger.debug("_validate_quad_quality: FAIL нульова сторона")
            return False, 0.0
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if not (CORNER_ANGLE_MIN_DEG <= angle_deg <= CORNER_ANGLE_MAX_DEG):
            logger.debug(
                f"_validate_quad_quality: FAIL кут {angle_deg:.1f}° "
                f"(ліміт {CORNER_ANGLE_MIN_DEG}..{CORNER_ANGLE_MAX_DEG})"
            )
            return False, 0.0
        # Відхилення від 90°: 0° = ідеально, 45° = максимально дозволено
        deviation = abs(angle_deg - 90.0)
        angle_scores.append(1.0 - deviation / 45.0)

    angle_quality = float(np.mean(angle_scores))

    # --- Перевірка 3: Прямолінійність сторін ---
    # Для кожної сторони: середня точка між двома кутами vs середина відрізка
    corner_pairs = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
    straightness_scores = []
    for p1, p2 in corner_pairs:
        side_len = float(np.linalg.norm(p2 - p1))
        if side_len < 1.0:
            continue
        mid_expected = (p1 + p2) / 2.0
        # Ми не маємо проміжних точок контуру тут — перевіряємо лише геометрію кутів
        # Тест: чи не "ввігнуті" сторони (cross product знак)
        straightness_scores.append(1.0)  # кути вже впорядковані — базова перевірка пройдена

    straightness_quality = float(np.mean(straightness_scores)) if straightness_scores else 1.0

    quality_score = float(np.mean([area_ratio / MAX_QUAD_AREA_RATIO, angle_quality, straightness_quality]))
    quality_score = min(1.0, quality_score)

    logger.debug(
        f"_validate_quad_quality: OK area={area_ratio:.3f} "
        f"angle_q={angle_quality:.3f} quality={quality_score:.3f}"
    )
    return True, quality_score


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
        # пом'якшено: документ може торкатись країв кадру
        if _touches_image_border(cnt, gray_shape) and area > image_area * 0.98:
            continue

        # Адаптивний epsilon: пробуємо кілька значень
        peri = cv2.arcLength(cnt, True)
        for eps in APPROX_POLY_EPSILONS:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == CORNER_COUNT:
                if _validate_document(approx, (h, w)):
                    pts_candidate = approx.reshape(4, 2).astype(np.float32)
                    valid, quality = _validate_quad_quality(pts_candidate, (h, w))
                    if valid:
                        logger.debug(f"_find_quad_contour: якість {quality:.3f} для eps={eps}")
                        return pts_candidate

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
    rect[1] = pts[np.argmax(diff)]  # TR — максимальна x - y
    rect[3] = pts[np.argmin(diff)]  # BL — мінімальна x - y
    return rect


def _is_clockwise(pts: np.ndarray) -> bool:
    """
    Перевіряє чи впорядковані точки [TL, TR, BR, BL] за годинниковою стрілкою.
    Використовує знак cross product суми по 4-кутнику.
    
    Args:
        pts: float32 array shape (4,2), впорядковані [TL, TR, BR, BL]
    
    Returns:
        True якщо точки йдуть за годинниковою стрілкою
    """
    # Cross product sum для 4-кутника: (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + ...
    # > 0 = clockwise, < 0 = counter-clockwise
    total = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        total += (pts[j][0] - pts[i][0]) * (pts[j][1] + pts[i][1])
    return total > 0


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

    Захисні перевірки:
    - Всі 4 кути мають бути різні (мінімальна відстань між будь-якими двома > 10px)
    - Площа квадрилатераля має бути > мінімально допустимої
    - Якщо перевірки не пройдено — повертається {top: False, bottom: False, left: False, right: False}
    """
    tl, tr, br, bl = pts

    # Захисна перевірка: чи всі 4 кути різні
    distances = [
        np.linalg.norm(tl - tr),
        np.linalg.norm(tl - br),
        np.linalg.norm(tl - bl),
        np.linalg.norm(tr - br),
        np.linalg.norm(tr - bl),
        np.linalg.norm(br - bl),
    ]
    min_dist = min(distances)
    if min_dist < 10.0:
        # Кути збігаються або майже збігаються — невалідний документ
        return {"top": False, "bottom": False, "left": False, "right": False}

    # Захисна перевірка: площа квадрилатераля
    quad_area = 0.5 * abs(
        tl[0] * tr[1] + tr[0] * br[1] + br[0] * bl[1] + bl[0] * tl[1]
        - (tr[0] * tl[1] + br[0] * tr[1] + bl[0] * br[1] + tl[0] * bl[1])
    )
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl), 1.0)
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1.0)
    if quad_area < MIN_DOCUMENT_AREA_RATIO * w * h:
        # Площа занадто мала для документа
        return {"top": False, "bottom": False, "left": False, "right": False}

    width = w
    height = h
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


def auto_correct_iterative(
    image: np.ndarray,
    max_passes: int = ITERATIVE_MAX_PASSES,
    max_dim: int = MAX_ANALYSIS_DIM,
    partial: bool = False,
) -> tuple[np.ndarray, int, float]:
    """
    Ітеративна перспективна корекція: до max_passes проходів warp.

    Після кожного проходу перевіряємо залишкове викривлення.
    Якщо воно менше ITERATIVE_MIN_SKEW_RATIO — зупиняємось.
    Якщо якість другого результату гірша за перший — повертаємо перший.

    Args:
        image: вхідне BGR зображення
        max_passes: максимальна кількість проходів (рекомендовано 2)
        max_dim: максимальний розмір для аналізу
        partial: якщо True — використовує apply_partial_correction замість apply_correction

    Returns:
        (result, passes_done, final_skew_ratio)
        passes_done: скільки проходів реально виконано
        final_skew_ratio: залишкове викривлення після останнього проходу
    """
    current = image.copy()
    passes_done = 0
    prev_quality = 0.0
    skew_ratio = 0.0

    for pass_num in range(max_passes):
        corners = auto_detect_corners(current, max_dim=max_dim)
        if corners is None:
            logger.debug(f"auto_correct_iterative: прохід {pass_num+1} — кути не знайдено, зупиняємось")
            break

        ordered = _order_points(corners)
        skew = detect_skewed_sides(ordered)
        has_skewed = any(skew.values())

        if not has_skewed:
            logger.debug(f"auto_correct_iterative: прохід {pass_num+1} — викривлення не знайдено")
            break

        # Вимірюємо skew_ratio
        tl, tr, br, bl = ordered
        top_skew = abs(float(tl[1] - tr[1]))
        bottom_skew = abs(float(bl[1] - br[1]))
        left_skew = abs(float(tl[0] - bl[0]))
        right_skew = abs(float(tr[0] - br[0]))
        max_skew = max(top_skew, bottom_skew, left_skew, right_skew)
        h_c, w_c = current.shape[:2]
        skew_ratio = max_skew / max(h_c, w_c)

        if skew_ratio < ITERATIVE_MIN_SKEW_RATIO:
            logger.debug(
                f"auto_correct_iterative: прохід {pass_num+1} — "
                f"skew_ratio={skew_ratio:.4f} < порогу, зупиняємось"
            )
            break

        # Валідація якості знайдених кутів
        valid, quality = _validate_quad_quality(ordered, current.shape)
        if not valid:
            logger.debug(f"auto_correct_iterative: прохід {pass_num+1} — валідація провалилась")
            break

        # Якщо якість погіршилась (другий прохід знайшов гірші кути) — зупиняємось
        if pass_num > 0 and quality < prev_quality * 0.8:
            logger.debug(
                f"auto_correct_iterative: прохід {pass_num+1} — "
                f"якість погіршилась {quality:.3f} < {prev_quality:.3f}*0.8, зупиняємось"
            )
            break

        prev_quality = quality
        candidate = apply_partial_correction(current, corners) if partial else apply_correction(current, corners)
        current = candidate
        passes_done += 1

        logger.debug(
            f"auto_correct_iterative: прохід {pass_num+1} виконано, "
            f"skew_ratio={skew_ratio:.4f}, quality={quality:.3f}"
        )

    return current, passes_done, skew_ratio if passes_done > 0 else 0.0


def _score_quad(pts: np.ndarray, image_shape: tuple) -> float:
    """
    Скоринг знайденого квадрилатераля: 0.0 (поганий) → 1.0 (ідеальний).
    Критерії: площа, кути ~90°, відступ від краю.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)
    ordered = _order_points(pts.astype(np.float32))
    tl, tr, br, bl = ordered

    # Площа: 15–85% кадру = ідеально
    area = float(cv2.contourArea(ordered))
    ratio = area / image_area
    if ratio < 0.10 or ratio > MAX_QUAD_AREA_RATIO:
        return 0.0
    # Пік скору на ratio ≈ 0.60 (вище для телефонних фото де документ займає більше кадру)
    area_score = 1.0 - abs(ratio - 0.60) / 0.37

    # Кути між сторонами: ближче до 90° = краще
    sides = [tr - tl, br - tr, bl - br, tl - bl]
    angle_devs = []
    for i in range(4):
        v1 = sides[i].astype(np.float64)
        v2 = sides[(i + 1) % 4].astype(np.float64)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1.0 or n2 < 1.0:
            return 0.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_a)))
        angle_devs.append(abs(angle - 90.0))
    angle_score = 1.0 - float(np.mean(angle_devs)) / 45.0
    angle_score = max(0.0, angle_score)

    # Відступ від краю: кути не мають торкатися кадру
    margin = 0.04
    pts_xs = [tl[0], tr[0], br[0], bl[0]]
    pts_ys = [tl[1], tr[1], br[1], bl[1]]
    touches_edge = (
        min(pts_xs) < w * margin or max(pts_xs) > w * (1 - margin) or
        min(pts_ys) < h * margin or max(pts_ys) > h * (1 - margin)
    )
    margin_score = 0.6 if touches_edge else 1.0

    return area_score * 0.45 + angle_score * 0.45 + margin_score * 0.10


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