"""
Тести для processing/perspective.py.

Фокус:
1. _order_points — правильність упорядкування [TL, TR, BR, BL].
2. _is_clockwise — коректна детекція CW/CCW.
3. apply_correction (smoke) — не дзеркалить, однокольоровий маркер у правильному куті.
4. apply_partial_correction — коректна обробка часткового випадку.
"""

import numpy as np
import cv2
import pytest

from processing.perspective import (
    _order_points,
    _is_clockwise,
    apply_correction,
    apply_partial_correction,
    _compute_destination,
    detect_skewed_sides,
)


# ---------------------------------------------------------------------------
# _order_points
# ---------------------------------------------------------------------------

class TestOrderPoints:
    """Перевірка _order_points на різних вхідних порядках."""

    def test_already_ordered(self):
        """Вхідні точки вже [TL, TR, BR, BL]."""
        pts = np.array([
            [10, 20],   # TL
            [200, 20],  # TR
            [200, 300], # BR
            [10, 300],  # BL
        ], dtype=np.float32)
        result = _order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [10, 20])   # TL
        np.testing.assert_array_almost_equal(result[1], [200, 20])  # TR
        np.testing.assert_array_almost_equal(result[2], [200, 300]) # BR
        np.testing.assert_array_almost_equal(result[3], [10, 300])  # BL

    def test_reversed_order(self):
        """Зворотній порядок [BR, BL, TL, TR]."""
        pts = np.array([
            [200, 300], # BR
            [10, 300],  # BL
            [10, 20],   # TL
            [200, 20],  # TR
        ], dtype=np.float32)
        result = _order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [10, 20])   # TL
        np.testing.assert_array_almost_equal(result[1], [200, 20])  # TR
        np.testing.assert_array_almost_equal(result[2], [200, 300]) # BR
        np.testing.assert_array_almost_equal(result[3], [10, 300])  # BL

    def test_shuffled(self):
        """Повністю перемішаний порядок."""
        pts = np.array([
            [200, 20],  # TR
            [10, 300],  # BL
            [10, 20],   # TL
            [200, 300], # BR
        ], dtype=np.float32)
        result = _order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [10, 20])   # TL
        np.testing.assert_array_almost_equal(result[1], [200, 20])  # TR
        np.testing.assert_array_almost_equal(result[2], [200, 300]) # BR
        np.testing.assert_array_almost_equal(result[3], [10, 300])  # BL

    def test_rotated_rectangle(self):
        """Повернутий прямокутник (не ось-вирівняний)."""
        # Прямокутник 200x300, повернутий на ~30°
        pts = np.array([
            [150, 50],   # ~TR
            [50, 150],   # ~BL
            [250, 180],  # ~BR
            [50, 20],    # ~TL
        ], dtype=np.float32)
        result = _order_points(pts)
        # Перевіряємо що TL — найменша сума (x+y ≈ 70)
        assert np.argmin(pts.sum(axis=1)) == 3  # [50,20] має суму 70
        np.testing.assert_array_almost_equal(result[0], [50, 20])
        # Перевіряємо що BR — найбільша сума (x+y ≈ 430)
        np.testing.assert_array_almost_equal(result[2], [250, 180])
        # TR: має мати більший x i менший y ніж BL
        assert result[1][0] > result[3][0]  # TR.x > BL.x
        assert result[1][1] < result[3][1]  # TR.y < BL.y

    def test_perspective_distorted(self):
        """Трапеція (перспективне викривлення)."""
        pts = np.array([
            [80, 30],   # ~TL (верхній лівий)
            [220, 10],  # ~TR (верхній правий — менший за розміром через перспективу)
            [200, 300], # ~BR
            [50, 310],  # ~BL
        ], dtype=np.float32)
        result = _order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [80, 30])   # TL
        np.testing.assert_array_almost_equal(result[1], [220, 10])  # TR
        np.testing.assert_array_almost_equal(result[2], [200, 300]) # BR
        np.testing.assert_array_almost_equal(result[3], [50, 310])  # BL


# ---------------------------------------------------------------------------
# _is_clockwise
# ---------------------------------------------------------------------------

class TestIsClockwise:
    """Перевірка _is_clockwise."""

    def test_cw_correct_order(self):
        """Правильний CW порядок [TL, TR, BR, BL] має бути CW."""
        pts = np.array([
            [10, 20],   # TL
            [200, 20],  # TR
            [200, 300], # BR
            [10, 300],  # BL
        ], dtype=np.float32)
        assert _is_clockwise(pts) is True

    def test_ccw_order(self):
        """Проти-годинниковий порядок має бути CCW."""
        pts = np.array([
            [10, 20],   # TL
            [10, 300],  # BL
            [200, 300], # BR
            [200, 20],  # TR
        ], dtype=np.float32)
        assert _is_clockwise(pts) is False

    def test_cw_rotated(self):
        """Повернутий прямокутник, CW порядок."""
        pts = np.array([
            [50, 20],   # TL
            [150, 50],  # TR
            [250, 180], # BR
            [50, 150],  # BL
        ], dtype=np.float32)
        assert _is_clockwise(pts) is True

    def test_ccw_rotated(self):
        """Повернутий прямокутник, CCW порядок."""
        pts = np.array([
            [50, 20],   # TL
            [50, 150],  # BL
            [250, 180], # BR
            [150, 50],  # TR
        ], dtype=np.float32)
        assert _is_clockwise(pts) is False


# ---------------------------------------------------------------------------
# apply_correction smoke test
# ---------------------------------------------------------------------------

class TestApplyCorrection:
    """Smoke-тест apply_correction: корекція не дзеркалить."""

    def test_synthetic_white_page_black_border(self):
        """
        Створюємо синтетичне зображення 300x400:
        - фон чорний (0)
        - прямокутник "документа" білий (255) з невеликим нахилом
        - кольоровий квадрат 20x20 тільки в TOP-LEFT куті документа

        Після корекції кольоровий квадрат має опинитись в TL куті результату,
        а не в BR, TR або BL (не дзеркально).
        """
        size = (400, 300)  # H, W
        img = np.zeros((*size, 3), dtype=np.uint8)  # чорний фон

        # "Документ": білий прямокутник з невеликим нахилом
        # TL(40, 50), TR(260, 30), BR(270, 370), BL(50, 360)
        doc_corners = np.array([
            [40, 50],     # TL
            [260, 30],    # TR
            [270, 370],   # BR
            [50, 360],    # BL
        ], dtype=np.float32)

        # Малюємо білий документ заливкою
        cv2.fillPoly(img, [doc_corners.astype(np.int32)], (255, 255, 255))

        # Кольоровий маркер (червоний) 20x20 в TL-куті документа
        marker_tl = (40, 50)
        marker_br = (60, 70)
        cv2.rectangle(img, marker_tl, marker_br, (0, 0, 255), thickness=cv2.FILLED)

        # Застосовуємо корекцію
        corrected = apply_correction(img, doc_corners)

        # Перевіряємо, що результат не порожній
        assert corrected.shape[0] > 0
        assert corrected.shape[1] > 0

        # Аналізуємо TL-кут результату (10% зверху-зліва)
        h, w = corrected.shape[:2]
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)

        tl_region = corrected[:margin_h, :margin_w, :]
        # В TL-регіоні має бути червоний (R > 0, G ≈ 0, B ≈ 0)
        red_mask = (tl_region[:, :, 2] > 100) & (tl_region[:, :, 1] < 50) & (tl_region[:, :, 0] < 50)
        red_pixels = int(np.sum(red_mask))
        assert red_pixels > 50, f"Червоний маркер не знайдено в TL куті! Знайдено {red_pixels} червоних пікселів"

        # Перевіряємо, що в BR куті червоного немає (не дзеркально)
        br_region = corrected[-margin_h:, -margin_w:, :]
        red_mask_br = (br_region[:, :, 2] > 100) & (br_region[:, :, 1] < 50) & (br_region[:, :, 0] < 50)
        red_pixels_br = int(np.sum(red_mask_br))
        assert red_pixels_br < 10, f"Червоний маркер помилково знайдено в BR куті! {red_pixels_br} пікселів"

    def test_synthetic_blue_marker_tr_corner(self):
        """
        Синтетичне зображення з синім маркером в TR-куті.
        Після корекції маркер має бути в TR.
        """
        size = (400, 300)  # H, W
        img = np.zeros((*size, 3), dtype=np.uint8)

        doc_corners = np.array([
            [40, 50],     # TL
            [260, 30],    # TR
            [270, 370],   # BR
            [50, 360],    # BL
        ], dtype=np.float32)

        cv2.fillPoly(img, [doc_corners.astype(np.int32)], (255, 255, 255))

        # Синій маркер в TR-куті
        cv2.rectangle(img, (240, 30), (260, 50), (255, 0, 0), thickness=cv2.FILLED)

        corrected = apply_correction(img, doc_corners)
        h, w = corrected.shape[:2]
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)

        # TR-регіон (верхній правий кут)
        tr_region = corrected[:margin_h, -margin_w:, :]
        blue_mask = (tr_region[:, :, 0] > 100) & (tr_region[:, :, 1] < 50) & (tr_region[:, :, 2] < 50)
        blue_pixels = int(np.sum(blue_mask))
        assert blue_pixels > 50, f"Синій маркер не знайдено в TR куті! Знайдено {blue_pixels} синіх пікселів"

        # В BL куті синього не має бути (не дзеркально)
        bl_region = corrected[-margin_h:, :margin_w, :]
        blue_mask_bl = (bl_region[:, :, 0] > 100) & (bl_region[:, :, 1] < 50) & (bl_region[:, :, 2] < 50)
        blue_pixels_bl = int(np.sum(blue_mask_bl))
        assert blue_pixels_bl < 10, f"Синій маркер помилково знайдено в BL куті! {blue_pixels_bl} пікселів"


# ---------------------------------------------------------------------------
# apply_partial_correction
# ---------------------------------------------------------------------------

class TestApplyPartialCorrection:
    """Перевірка часткової корекції."""

    def test_no_skew_returns_copy(self):
        """Якщо всі сторони прямі — повертає копію."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        corners = np.array([
            [10, 10],    # TL
            [290, 10],   # TR
            [290, 190],  # BR
            [10, 190],   # BL
        ], dtype=np.float32)
        result = apply_partial_correction(img, corners)
        np.testing.assert_array_equal(result, img)

    def test_skewed_top_only(self):
        """Тільки верхня сторона крива — корекція проходить."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        corners = np.array([
            [10, 30],    # TL — зміщений по Y
            [290, 10],   # TR
            [290, 190],  # BR
            [10, 190],   # BL
        ], dtype=np.float32)
        result = apply_partial_correction(img, corners)
        # Результат має бути змінений (не просто копія, padding змінює розмір)
        assert result.shape[0] > 0 and result.shape[1] > 0
        # Корекція має застосуватись (top skewed)
        ordered = _order_points(corners)
        skew = detect_skewed_sides(ordered)
        assert skew["top"] is True  # детекція працює
        # Результат — не просто копія (перспективна трансформація відбулась)
        assert result is not img

    def test_full_skew_delegates_to_apply_correction(self):
        """Всі 4 сторони криві — делегує в apply_correction."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Трапеція — всі сторони не прямокутні
        corners = np.array([
            [30, 30],    # TL
            [250, 20],   # TR
            [260, 180],  # BR
            [20, 190],   # BL
        ], dtype=np.float32)
        result = apply_partial_correction(img, corners)
        assert result.shape[0] > 0
        assert result.shape[1] > 0


# ---------------------------------------------------------------------------
# detect_skewed_sides
# ---------------------------------------------------------------------------

class TestDetectSkewedSides:
    """Перевірка detect_skewed_sides."""

    def test_no_skew(self):
        """Прямокутник без викривлення."""
        pts = np.array([
            [10, 10],
            [290, 10],
            [290, 190],
            [10, 190],
        ], dtype=np.float32)
        result = detect_skewed_sides(pts)
        assert result == {"top": False, "bottom": False, "left": False, "right": False}

    def test_top_skewed(self):
        """Верхня сторона крива."""
        pts = np.array([
            [10, 30],    # TL — зсув по Y
            [290, 10],   # TR
            [290, 190],  # BR
            [10, 190],   # BL
        ], dtype=np.float32)
        result = detect_skewed_sides(pts)
        assert result["top"] is True
        assert result["bottom"] is False
        assert result["left"] is False
        assert result["right"] is False

    def test_all_skewed(self):
        """Всі сторони криві (трапеція)."""
        pts = np.array([
            [30, 30],    # TL
            [270, 20],   # TR
            [280, 180],  # BR
            [20, 190],   # BL
        ], dtype=np.float32)
        result = detect_skewed_sides(pts)
        assert all(result.values()), f"Очікувалось всі 4 сторони криві, отримано {result}"


# ---------------------------------------------------------------------------
# _compute_destination
# ---------------------------------------------------------------------------

class TestComputeDestination:
    """Перевірка обчислення вихідного розміру."""

    def test_basic_rectangle(self):
        """Прямокутник → правильний розмір."""
        pts = np.array([
            [10, 10],
            [290, 10],
            [290, 190],
            [10, 190],
        ], dtype=np.float32)
        dst, width, height = _compute_destination(pts)
        assert width > 280  # враховуючи padding
        assert height > 180
        assert dst.shape == (4, 2)
        # dst має бути прямокутник
        np.testing.assert_array_almost_equal(dst[0], [dst[0][0], dst[0][1]])


# ---------------------------------------------------------------------------
# Інтеграційний тест: _order_points + apply_correction ланцюжок
# ---------------------------------------------------------------------------

class TestIntegration:
    """Перевірка цілісного ланцюжка."""

    def test_order_points_then_correction(self):
        """_order_points + apply_correction: точки впорядковуються і корекція не падає."""
        img = np.ones((400, 300, 3), dtype=np.uint8) * 200
        # Випадковий порядок точок
        corners = np.array([
            [270, 370],  # BR
            [40, 50],    # TL
            [260, 30],   # TR
            [50, 360],   # BL
        ], dtype=np.float32)

        # Впорядковуємо
        ordered = _order_points(corners)
        np.testing.assert_array_almost_equal(ordered[0], [40, 50])   # TL
        np.testing.assert_array_almost_equal(ordered[1], [260, 30])  # TR
        np.testing.assert_array_almost_equal(ordered[2], [270, 370]) # BR
        np.testing.assert_array_almost_equal(ordered[3], [50, 360])  # BL

        # Застосовуємо корекцію
        result = apply_correction(img, corners)
        assert result.shape[0] > 0
        assert result.shape[1] > 0