"""
Юніт-тести для PRIO 2 — часткова корекція перспективи.
Перевіряє detect_skewed_sides, apply_partial_correction, auto_correct_partial.
"""

import numpy as np
import pytest
import cv2

from processing import perspective


# ---------------------------------------------------------------------------
# detect_skewed_sides
# ---------------------------------------------------------------------------

class TestDetectSkewedSides:
    def test_all_straight(self):
        """Ідеальний прямокутник — жодна сторона не крива."""
        pts = np.array([
            [0, 0],        # TL
            [400, 0],      # TR
            [400, 300],    # BR
            [0, 300],      # BL
        ], dtype=np.float32)
        skew = perspective.detect_skewed_sides(pts)
        assert not any(skew.values()), f"очікувалось всі прямі, отримано {skew}"

    def test_top_skewed(self):
        """Верхня сторона крива."""
        pts = np.array([
            [0, 10],       # TL — зміщено вниз
            [400, 0],      # TR — на місці
            [400, 300],    # BR
            [0, 300],      # BL
        ], dtype=np.float32)
        skew = perspective.detect_skewed_sides(pts)
        assert bool(skew["top"]) is True
        assert bool(skew["bottom"]) is False
        assert bool(skew["left"]) is False
        assert bool(skew["right"]) is False

    def test_left_skewed(self):
        """Ліва сторона крива (зміщення 15px > порогу 3% = 12px)."""
        pts = np.array([
            [15, 0],       # TL — зміщено вправо на 15px
            [400, 0],      # TR
            [400, 300],    # BR
            [0, 300],      # BL
        ], dtype=np.float32)
        skew = perspective.detect_skewed_sides(pts)
        assert bool(skew["top"]) is False
        assert bool(skew["left"]) is True

    def test_top_bottom_skewed(self):
        """Верх і низ криві."""
        pts = np.array([
            [0, 10],       # TL
            [400, 0],      # TR
            [400, 290],    # BR
            [0, 300],      # BL
        ], dtype=np.float32)
        skew = perspective.detect_skewed_sides(pts)
        assert bool(skew["top"]) is True
        assert bool(skew["bottom"]) is True

    def test_all_skewed(self):
        """Всі 4 сторони криві (зміщення 25-30px > порогу 3% = ~10px)."""
        pts = np.array([
            [40, 30],      # TL
            [360, 5],      # TR
            [330, 270],    # BR
            [10, 295],     # BL
        ], dtype=np.float32)
        skew = perspective.detect_skewed_sides(pts)
        assert all(skew.values())

    def test_small_skew_below_threshold(self):
        """Дуже мале відхилення не вважається кривим (< 3% порогу)."""
        pts = np.array([
            [0, 2],        # TL — відхилення 2px при висоті 1000px = 0.2%
            [400, 0],      # TR
            [400, 1000],   # BR
            [0, 1000],     # BL
        ], dtype=np.float32)
        skew = perspective.detect_skewed_sides(pts)
        assert bool(skew["top"]) is False  # 2px < 12px (3% of 400)


# ---------------------------------------------------------------------------
# apply_partial_correction
# ---------------------------------------------------------------------------

class TestApplyPartialCorrection:
    @pytest.fixture
    def image(self):
        """Зображення 400x300."""
        return np.ones((300, 400, 3), dtype=np.uint8) * 200

    def test_no_skew_returns_copy(self, image):
        """Якщо жодна сторона не крива — повертає копію."""
        corners = np.array([
            [0, 0], [400, 0], [400, 300], [0, 300]
        ], dtype=np.float32)
        result = perspective.apply_partial_correction(image, corners)
        assert np.array_equal(image, result)
        assert result is not image

    def test_all_skewed_delegates_to_full(self, image):
        """Всі 4 сторони криві — делегує в apply_correction."""
        corners = np.array([
            [10, 10], [390, 5], [395, 295], [5, 290]
        ], dtype=np.float32)
        # Виклик має спрацювати без помилок і повернути масив
        result = perspective.apply_partial_correction(image, corners)
        assert result.shape[-1] == 3
        assert result.dtype == np.uint8

    def test_partial_skew_output_shape(self, image):
        """Крива верхня сторона — результат має коректну форму."""
        corners = np.array([
            [10, 10],      # TL — зміщено
            [380, 0],      # TR
            [380, 300],    # BR
            [10, 300],     # BL
        ], dtype=np.float32)
        result = perspective.apply_partial_correction(image, corners)
        assert result.ndim == 3
        assert result.shape[-1] == 3
        assert result.dtype == np.uint8

    def test_partial_immutable(self, image):
        """Вхідне зображення не змінюється."""
        before = image.copy()
        corners = np.array([
            [10, 10], [380, 0], [380, 300], [10, 300]
        ], dtype=np.float32)
        _ = perspective.apply_partial_correction(image, corners)
        assert np.array_equal(image, before)


# ---------------------------------------------------------------------------
# auto_correct_partial
# ---------------------------------------------------------------------------

class TestAutoCorrectPartial:
    def test_detect_and_partial_correct(self):
        """Авто-детект + часткова корекція на синтетичному зображенні."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200
        cv2.rectangle(img, (50, 30), (550, 370), (50, 50, 50), -1)
        result, found = perspective.auto_correct_partial(img)
        assert found is True
        assert result.ndim == 3
        assert result.dtype == np.uint8

    def test_no_document_returns_copy(self):
        """Без документа (рівномірний фон) — повертає копію."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        result, found = perspective.auto_correct_partial(img)
        assert found is False
        assert np.array_equal(img, result)
        assert result is not img
