"""
Перспективна корекція документа.
Авто: знаходить 4 кути документа через контури.
Ручна: приймає 4 точки від користувача.
Не залежить від GUI модулів.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Публічний API
# ---------------------------------------------------------------------------

def auto_detect_corners(image: np.ndarray) -> np.ndarray | None:
    """
    Автоматично знаходить 4 кути документа.
    Повертає масив float32 shape (4,2) у порядку [TL, TR, BR, BL]
    або None якщо документ не знайдено.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed = _preprocess_for_detection(gray)
    contour = _find_document_contour(preprocessed)
    if contour is None:
        return None
    pts = _order_points(contour.reshape(4, 2).astype(np.float32))
    return pts


def apply_correction(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Виконує перспективну трансформацію за 4 кутами.
    corners: float32 array shape (4,2), порядок [TL, TR, BR, BL].
    Повертає випрямлений BGR uint8.
    """
    # Точки вже у правильному порядку [TL, TR, BR, BL] від GUI —
    # НЕ переупорядковуємо, бо це ламає ручну корекцію
    pts = corners.astype(np.float32)
    dst, width, height = _compute_destination(pts)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def auto_correct(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Зручна обгортка: авто-детект + корекція в одному виклику.
    Повертає (результат, True) якщо документ знайдено,
    або (оригінал, False) якщо ні.
    """
    corners = auto_detect_corners(image)
    if corners is None:
        return image.copy(), False
    return apply_correction(image, corners), True


# ---------------------------------------------------------------------------
# Внутрішні функції
# ---------------------------------------------------------------------------

def _preprocess_for_detection(gray: np.ndarray) -> np.ndarray:
    """Підготовка grayscale для пошуку контурів."""
    # Розмиття для зменшення шуму
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Адаптивний поріг — добре працює при нерівномірному освітленні
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    # Морфологія — з'єднуємо розірвані краї
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed


def _find_document_contour(binary: np.ndarray) -> np.ndarray | None:
    """
    Знаходить найбільший 4-кутний контур (документ).
    Повертає contour shape (4,1,2) або None.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Сортуємо за площею — найбільший контур першим
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = binary.shape[0] * binary.shape[1]

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        # Документ займає щонайменше 10% площі зображення
        if area < image_area * 0.10:
            break

        # Апроксимуємо контур до полігону
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядковує 4 точки: [TL, TR, BR, BL].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
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
    """
    tl, tr, br, bl = pts

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    dst = np.array([
        [0,         0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0,         height - 1],
    ], dtype=np.float32)

    return dst, width, height
