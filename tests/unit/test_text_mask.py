"""
Юніт-тести для processing/text_mask.py.
"""

import numpy as np
import cv2
import pytest
from processing import text_mask


class TestTextRegionMask:
    def test_output_shape_and_type(self):
        """Перевіряє, що маска має правильну форму та тип."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = text_mask.text_region_mask(img)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_uniform_image_no_text(self):
        """Однотонне зображення без тексту має маску ~0."""
        img = np.ones((50, 50), dtype=np.uint8) * 128
        mask = text_mask.text_region_mask(img)
        text_pixels = np.count_nonzero(mask)
        # Може бути трохи шуму, але не більше 5%
        assert text_pixels < 50 * 50 * 0.05, f"Text pixels: {text_pixels}"

    def test_image_with_text_detects_something(self):
        """Зображення з текстом (темні пікселі на світлому) має ненульову маску."""
        img = np.ones((100, 100), dtype=np.uint8) * 220
        # Малюємо кілька "літер"
        img[30:40, 30:35] = 30
        img[30:40, 40:45] = 30
        img[50:60, 50:55] = 30
        mask = text_mask.text_region_mask(img)
        text_pixels = np.count_nonzero(mask)
        assert text_pixels > 0, f"Text pixels: {text_pixels}"


class TestUniformAreaMask:
    def test_output_shape_and_type(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = text_mask.uniform_area_mask(img)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8

    def test_uniform_image_all_uniform(self):
        """Однотонне зображення майже вся маска = 255."""
        img = np.ones((64, 64), dtype=np.uint8) * 128
        mask = text_mask.uniform_area_mask(img)
        uniform_pixels = np.count_nonzero(mask)
        # > 90% має бути "однотонним"
        assert uniform_pixels > 64 * 64 * 0.9, f"Uniform pixels: {uniform_pixels}"

    def test_noisy_image_few_uniform(self):
        """Шумне зображення майже без однотонних областей."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = text_mask.uniform_area_mask(img)
        uniform_pixels = np.count_nonzero(mask)
        # На шумному зображенні однотонних блоків мало
        assert uniform_pixels < 64 * 64 * 0.5


class TestHDRApplyAdaptive:
    """Тести для hdr.apply_adaptive (інтеграція text_mask + hdr)."""

    def test_adaptive_with_mask_returns_different_result(self):
        """apply_adaptive з маскою дає інший результат, ніж без маски."""
        from processing import hdr as hdr_module
        np.random.seed(42)
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = text_mask.text_region_mask(gray)
        res_adaptive = hdr_module.apply_adaptive(img, strength=0.5, text_mask=mask)
        res_normal = hdr_module.apply(img, strength=0.5)
        # Хоча б один піксель відрізняється
        assert np.any(res_adaptive != res_normal)

    def test_adaptive_without_mask_equals_apply(self):
        """apply_adaptive без text_mask = apply."""
        from processing import hdr as hdr_module
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res1 = hdr_module.apply_adaptive(img, strength=0.5)
        res2 = hdr_module.apply(img, strength=0.5)
        assert np.array_equal(res1, res2)

    def test_adaptive_zero_strength_returns_copy(self):
        """strength=0 повертає копію без змін."""
        from processing import hdr as hdr_module
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res = hdr_module.apply_adaptive(img, strength=0.0)
        assert np.array_equal(img, res)
        assert res is not img

    def test_adaptive_preserves_dtype(self):
        """Результат apply_adaptive завжди uint8."""
        from processing import hdr as hdr_module
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = text_mask.text_region_mask(gray)
        res = hdr_module.apply_adaptive(img, strength=0.5, text_mask=mask)
        assert res.dtype == np.uint8


class TestAutofixAdaptiveHDR:
    """Інтеграційний тест: autofix.apply з adaptive_hdr=True."""

    def test_autofix_adaptive_hdr_works(self):
        """autofix.apply з adaptive_hdr=True не падає."""
        from processing import autofix
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        res = autofix.apply(img, sharpen_strength=0.0, hdr_strength=0.5, use_hdr=True, adaptive_hdr=True)
        assert res.dtype == np.uint8
        assert res.shape == img.shape