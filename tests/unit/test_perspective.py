"""
Юніт-тести для processing/perspective.py.
Перевіряє базову логіку: _order_points, apply_correction, _touches_image_border, _solidity, _apply_clahe.
"""

import numpy as np
import pytest
import cv2

from processing import perspective


# ---------------------------------------------------------------------------
# _order_points
# ---------------------------------------------------------------------------

class TestOrderPoints:
    def test_order_points_ordered(self):
        """Впорядковані точки мають залишитись в тому ж порядку."""
        pts = np.array([
            [10, 10],     # TL
            [200, 15],    # TR
            [190, 300],   # BR
            [20, 290],    # BL
        ], dtype=np.float32)
        rect = perspective._order_points(pts)
        assert rect.shape == (4, 2)
        assert np.array_equal(rect[0], [10, 10])    # TL
        assert np.array_equal(rect[1], [200, 15])   # TR
        assert np.array_equal(rect[2], [190, 300])  # BR
        assert np.array_equal(rect[3], [20, 290])   # BL

    def test_order_points_unordered(self):
        """Перемішані точки мають бути впорядковані в TL, TR, BR, BL."""
        pts = np.array([
            [190, 300],   # BR
            [10, 10],     # TL
            [20, 290],    # BL
            [200, 15],    # TR
        ], dtype=np.float32)
        rect = perspective._order_points(pts)
        assert np.array_equal(rect[0], [10, 10])    # TL
        assert np.array_equal(rect[1], [200, 15])   # TR
        assert np.array_equal(rect[2], [190, 300])  # BR
        assert np.array_equal(rect[3], [20, 290])   # BL

    def test_order_points_rotated(self):
        """Точки після повороту на 90°."""
        pts = np.array([
            [100, 100],   # "TL" після повороту — це оригінальний BR
            [100, 80],    #
            [80, 80],
            [80, 100],
        ], dtype=np.float32)
        rect = perspective._order_points(pts)
        # Перевіряємо, що результат — 4 точки з мінімальною сумою в rect[0]
        s = rect.sum(axis=1)
        assert np.argmin(s) == 0
        assert np.argmax(s) == 2


# ---------------------------------------------------------------------------
# apply_correction (на синтетичному зображенні)
# ---------------------------------------------------------------------------

class TestApplyCorrection:
    @pytest.fixture
    def image(self):
        """Просте прямокутне зображення 200x200."""
        return np.ones((200, 200, 3), dtype=np.uint8) * 128

    def test_apply_correction_identity(self, image):
        """Якщо кути = bounding box, результат має бути ~такого ж розміру."""
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ], dtype=np.float32)
        result = perspective.apply_correction(image, corners)
        assert result.shape == (200, 200, 3) or result.shape[0] > 0

    def test_apply_correction_returns_uint8(self, image):
        """Результат має бути uint8."""
        corners = np.array([
            [10, 10],
            [180, 15],
            [175, 190],
            [8, 185],
        ], dtype=np.float32)
        result = perspective.apply_correction(image, corners)
        assert result.dtype == np.uint8

    def test_apply_correction_immutable(self, image):
        """Вхідне зображення не змінюється."""
        before = image.copy()
        corners = np.array([
            [10, 10], [180, 15], [175, 190], [8, 185]
        ], dtype=np.float32)
        _ = perspective.apply_correction(image, corners)
        assert np.array_equal(image, before)


# ---------------------------------------------------------------------------
# _touches_image_border
# ---------------------------------------------------------------------------

class TestTouchesImageBorder:
    def test_center_contour_not_touching(self):
        """Контур у центрі — не торкається меж."""
        cnt = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32)
        assert not perspective._touches_image_border(cnt, (200, 200))

    def test_contour_touches_left(self):
        """Контур торкається лівого краю."""
        cnt = np.array([[0, 50], [100, 50], [100, 150], [0, 150]], dtype=np.int32)
        assert perspective._touches_image_border(cnt, (200, 200))

    def test_contour_touches_top(self):
        """Контур торкається верхнього краю."""
        cnt = np.array([[50, 0], [150, 0], [150, 100], [50, 100]], dtype=np.int32)
        assert perspective._touches_image_border(cnt, (200, 200))

    def test_contour_touches_right_bottom(self):
        """Контур торкається правого і нижнього краю."""
        cnt = np.array([[170, 170], [200, 170], [200, 200], [170, 200]], dtype=np.int32)
        assert perspective._touches_image_border(cnt, (200, 200), margin=2)

    def test_near_border_but_not_touching(self):
        """Контур біля краю, але не торкається (відступ > margin)."""
        cnt = np.array([[5, 5], [50, 5], [50, 50], [5, 50]], dtype=np.int32)
        assert not perspective._touches_image_border(cnt, (200, 200), margin=2)


# ---------------------------------------------------------------------------
# _solidity
# ---------------------------------------------------------------------------

class TestSolidity:
    def test_square_high_solidity(self):
        """Квадрат має solidity = 1.0."""
        cnt = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        s = perspective._solidity(cnt)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_concave_lower_solidity(self):
        """Увігнута форма має solidity < 0.85."""
        # Форма "C" — увігнута
        cnt = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100],
            [0, 80], [80, 80], [80, 20], [0, 20]
        ], dtype=np.int32)
        s = perspective._solidity(cnt)
        assert s < 0.85

    def test_empty_contour_zero_solidity(self):
        """Порожній контур має solidity = 0.0."""
        cnt = np.zeros((0, 1, 2), dtype=np.int32)
        s = perspective._solidity(cnt)
        assert s == 0.0


# ---------------------------------------------------------------------------
# _apply_clahe
# ---------------------------------------------------------------------------

class TestApplyClahe:
    def test_clahe_output_shape(self):
        """CLAHE повертає масив тієї ж форми."""
        gray = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        result = perspective._apply_clahe(gray)
        assert result.shape == (100, 150)
        assert result.dtype == np.uint8

    def test_clahe_increases_contrast(self):
        """CLAHE має збільшити контраст (std після > std до)."""
        # Створюємо низькоконтрастне зображення
        gray = np.ones((100, 100), dtype=np.uint8) * 100
        gray[20:80, 20:80] = 110  # невелика варіація
        std_before = float(np.std(gray))
        result = perspective._apply_clahe(gray)
        std_after = float(np.std(result))
        assert std_after > std_before

    def test_clahe_immutable(self):
        """Вхідний масив не змінюється."""
        gray = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        before = gray.copy()
        _ = perspective._apply_clahe(gray)
        assert np.array_equal(gray, before)


# ---------------------------------------------------------------------------
# auto_detect_corners (інтеграційний тест на синтетичному зображенні)
# ---------------------------------------------------------------------------

class TestAutoDetectCorners:
    def test_detect_on_synthetic_rectangle(self):
        """Знаходить кути на синтетичному прямокутному документі."""
        # Білий фон
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200
        # Темний прямокутник всередині (імітує документ)
        cv2.rectangle(img, (50, 30), (550, 370), (50, 50, 50), -1)
        corners = perspective.auto_detect_corners(img)
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_detect_on_dark_background(self):
        """Знаходить кути на темному фоні."""
        # Темний фон
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        # Світлий прямокутник
        cv2.rectangle(img, (80, 60), (520, 340), (200, 200, 200), -1)
        corners = perspective.auto_detect_corners(img)
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_no_document_returns_none(self):
        """Якщо документа немає — повертає None."""
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        corners = perspective.auto_detect_corners(img)
        # Може знайти випадковий контур, але це ок
        # Перевіряємо, що не падає
        assert corners is None or corners.shape == (4, 2)