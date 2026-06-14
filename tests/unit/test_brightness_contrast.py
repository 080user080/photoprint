"""
Юніт-тести для processing/brightness_contrast.py.
"""

import numpy as np
import pytest
from processing import brightness_contrast as bc


def _gray_bgr(h=100, w=100, val=128):
    """Допоміжна: однотонне BGR зображення."""
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestApplyBrightness:
    def test_zero_returns_copy(self):
        img = _gray_bgr()
        res = bc.apply_brightness(img, 0.0)
        assert np.array_equal(img, res)
        assert res is not img

    def test_positive_changes_pixels(self):
        img = _gray_bgr(val=100)
        res = bc.apply_brightness(img, 0.5)
        assert not np.array_equal(img, res)

    def test_negative_changes_pixels(self):
        img = _gray_bgr(val=200)
        res = bc.apply_brightness(img, -0.5)
        assert not np.array_equal(img, res)

    def test_output_range(self):
        img = _gray_bgr(val=100)
        res = bc.apply_brightness(img, 0.9)
        assert res.min() >= 0
        assert res.max() <= 255


class TestAutoBrightness:
    def test_returns_copy_if_range_small(self):
        """Якщо діапазон вже достатній — повертає копію."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        # Додаємо невелику варіацію
        img[0, 0] = [130, 130, 130]
        res = bc.auto_brightness(img, percentile_low=5.0, percentile_high=95.0)
        assert np.array_equal(img, res)
        assert res is not img

    def test_stretches_histogram(self):
        """Розтягує вузький діапазон."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img[10:40, 10:40] = 110
        res = bc.auto_brightness(img, percentile_low=5.0, percentile_high=95.0)
        # Після розтягування має бути ширший діапазон
        assert res.max() > 110 or res.min() < 100


class TestApplyContrast:
    def test_zero_returns_copy(self):
        img = _gray_bgr()
        res = bc.apply_contrast(img, 0.0)
        assert np.array_equal(img, res)
        assert res is not img

    def test_positive_increases_contrast(self):
        img = _gray_bgr(val=128)
        img[0:50, :] = 100  # темна половина
        std_before = float(np.std(img))
        res = bc.apply_contrast(img, 0.5)
        std_after = float(np.std(res))
        assert std_after >= std_before - 0.5  # дозволяємо невелику похибку

    def test_negative_decreases_contrast(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:40, 10:40] = 200
        std_before = float(np.std(img))
        res = bc.apply_contrast(img, -0.5)
        std_after = float(np.std(res))
        assert std_after < std_before


class TestToGrayscale:
    def test_returns_3_channel(self):
        img = _gray_bgr()
        res = bc.to_grayscale(img)
        assert res.shape == (100, 100, 3)
        assert res.dtype == np.uint8

    def test_all_channels_equal(self):
        """Після grayscale всі 3 канали однакові."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res = bc.to_grayscale(img)
        assert np.array_equal(res[:, :, 0], res[:, :, 1])
        assert np.array_equal(res[:, :, 1], res[:, :, 2])