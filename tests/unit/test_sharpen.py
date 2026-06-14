"""
Юніт-тести для processing/sharpen.py.
"""

import numpy as np
import pytest
from processing import sharpen


class TestApply:
    def test_zero_returns_copy(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        res = sharpen.apply(img, 0.0)
        assert np.array_equal(img, res)
        assert res is not img

    def test_positive_changes_image(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res = sharpen.apply(img, 0.5)
        assert not np.array_equal(img, res)

    def test_clips_strength_to_max(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res = sharpen.apply(img, 5.0)  # > MAX_STRENGTH=1.0
        assert res.dtype == np.uint8

    def test_output_range(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res = sharpen.apply(img, 0.8)
        assert res.min() >= 0
        assert res.max() <= 255


class TestMeasureSharpness:
    def test_blurred_lower_than_sharp(self):
        """Розмите зображення має менший variance, ніж різке."""
        sharp = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        import cv2
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        s_sharp = sharpen.measure_sharpness(sharp)
        s_blurred = sharpen.measure_sharpness(blurred)
        assert s_sharp > s_blurred

    def test_constant_image_returns_zero(self):
        """Однотонне зображення має variance ~0."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        v = sharpen.measure_sharpness(img)
        assert v < 1.0


class TestAutoApply:
    def test_sharp_image_no_change(self):
        """Дуже різке зображення не отримує додаткової різкості."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res, strength = sharpen.auto_apply(img, threshold=1.0)
        assert np.array_equal(img, res)
        assert strength == 0.0

    def test_blurry_image_gets_sharpened(self):
        """Розмите зображення отримує різкість > 0."""
        import cv2
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        res, strength = sharpen.auto_apply(blurred, threshold=80.0, max_strength=0.7)
        assert strength > 0.0
        assert not np.array_equal(blurred, res)