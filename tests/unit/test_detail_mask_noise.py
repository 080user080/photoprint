"""
Тести для processing/detail_map.py — адаптивний noise_floor.
Перевіряє, що на рівному сірому фоні маска ≈ 0.
"""

import numpy as np
import cv2
import pytest
from processing.detail_map import detail_mask, local_std_map


class TestDetailMaskNoise:
    def test_uniform_gray_mask_near_zero(self):
        """На рівному сірому фоні detail_mask ≈ 0."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask = detail_mask(gray)
        assert mask.max() < 0.05, f"Mask should be near 0 on uniform, got max={mask.max()}"
        assert mask.mean() < 0.01

    def test_noise_floor_raises_threshold(self):
        """З вищим noise_floor маска стає ще нижчою на рівному фоні."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask_default = detail_mask(gray)
        mask_high = detail_mask(gray, noise_floor=10.0)
        # З вищим noise_floor маска має бути <= ніж з дефолтним
        assert mask_high.max() <= mask_default.max() + 1e-6

    def test_texture_mask_high(self):
        """На текстурному/шумовому зображенні detail_mask > 0.5."""
        np.random.seed(42)
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = detail_mask(gray)
        assert mask.mean() > 0.5, f"Mask should be high on texture, got mean={mask.mean()}"

    def test_noise_floor_auto_computation(self):
        """Автоматичне обчислення noise_floor на рівному фоні ≈ 0."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask = detail_mask(gray)  # noise_floor=None — авто
        # На ідеально рівному фоні маска ≈ 0
        assert mask.max() < 0.05

    def test_noise_floor_from_noise(self):
        """Авто noise_floor вищий за явно малий → маска вища з малим noise_floor (ref_std нижчий)."""
        np.random.seed(42)
        noise = np.random.normal(128, 15, (100, 100)).astype(np.uint8)
        mask_without = detail_mask(noise)
        # Авто noise_floor ≈ max(p10*2, 3) = ~24 для цього шуму.
        # Явний noise_floor=50 вищий за авто, піднімає ref_std → маска нижча
        mask_with_high = detail_mask(noise, noise_floor=50.0)
        assert mask_with_high.mean() <= mask_without.mean()

    def test_noise_floor_lower_than_auto(self):
        """Явний noise_floor нижчий за авто → маска вища (ref_std нижчий)."""
        np.random.seed(42)
        noise = np.random.normal(128, 15, (100, 100)).astype(np.uint8)
        mask_auto = detail_mask(noise)
        # Авто noise_floor ≈ 24. Явний 10 нижчий → ref_std = max(p90, 10) = p90 ≈ 17
        # В результаті маска вища
        mask_low = detail_mask(noise, noise_floor=10.0)
        assert mask_low.mean() > mask_auto.mean()

    def test_detail_mask_immutable(self):
        """detail_mask не змінює вхідний масив."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        gray_copy = gray.copy()
        _ = detail_mask(gray)
        assert np.array_equal(gray, gray_copy)

    def test_gradient_image_mask(self):
        """Градієнтне зображення має маску > 0 на перепаді."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        for x in range(100):
            gray[:, x] = x * 255 // 99  # градієнт 0..255
        mask = detail_mask(gray)
        # На перепаді яскравості маска має бути > 0
        assert mask.mean() > 0.1