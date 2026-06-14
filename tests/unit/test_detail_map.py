"""
Unit-тесты для processing.detail_map.
"""

import cv2
import numpy as np
import pytest

from processing import detail_map


class TestLocalStdMap:
    """Тесты для local_std_map."""

    def test_output_shape_and_dtype(self):
        """Форма == входной, dtype == float32, диапазон [0, inf)."""
        gray = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        result = detail_map.local_std_map(gray)
        assert result.shape == (100, 200)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)

    def test_flat_image_returns_zero_std(self):
        """Полностью однотонное изображение дает std = 0 везде."""
        gray = np.full((50, 50), 128, dtype=np.uint8)
        result = detail_map.local_std_map(gray)
        assert np.max(result) < 1e-6

    def test_high_contrast_edge(self):
        """На границе черного/белого квадрата std > 0 (переход дает вариацию)."""
        gray = np.zeros((50, 50), dtype=np.uint8)
        gray[15:35, 15:35] = 255
        result = detail_map.local_std_map(gray)
        # На границе квадрата std должен быть заметно > 0
        edge_pixels = result[15, 15:35]  # горизонтальная граница
        assert np.any(edge_pixels > 20.0)


class TestDetailMask:
    """Тесты для detail_mask."""

    def test_flat_image_with_weak_noise_returns_low_mask(self):
        """
        На изображении 400x400 c заливкой 230 + Gaussian noise(std=1):
        detail_mask возвращает среднее < 0.5 (слабый шум без текстур).
        DETAIL_MIN_REF_STD=3.0 ограничивает ref_std снизу.
        """
        np.random.seed(42)
        gray = np.full((400, 400), 230, dtype=np.uint8)
        noise = np.random.normal(0, 1, (400, 400)).astype(np.int16)
        gray = np.clip(gray + noise, 0, 255).astype(np.uint8)

        mask = detail_map.detail_mask(gray)
        assert mask.dtype == np.float32
        assert mask.shape == (400, 400)
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
        # Среднее маски < 0.5 (слабый шум, ref_std ограничен DETAIL_MIN_REF_STD=3.0)
        assert np.mean(mask) < 0.5, f"Mean mask value: {np.mean(mask):.4f}"

    def test_textured_region_returns_high_mask(self):
        """
        На изображении с текстурой (шум с высоким std) на плоском фоне:
        detail_mask ≈ 1 в текстурированной области и ≈ 0 на плоском фоне.
        """
        np.random.seed(42)
        gray = np.full((200, 200), 200, dtype=np.uint8)  # светло-серый фон
        # Текстурированная область с высоким контрастом
        texture = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        gray[50:150, 50:150] = texture

        mask = detail_map.detail_mask(gray)

        # Внутри текстурированной области должно быть значительно > 0
        inner = mask[60:140, 60:140]
        assert np.mean(inner) > 0.50, f"Mean inner mask: {np.mean(inner):.4f}"

        # Снаружи (на плоском фоне) должно быть ~0
        outer = mask[:30, :30]
        assert np.mean(outer) < 0.1, f"Mean outer mask: {np.mean(outer):.4f}"

        # Проверяем плавность перехода: соседние пиксели на границе не должны
        # отличаться более чем на ~0.06 (нет блочных артефактов)
        border_row = mask[50, 40:60]
        diffs = np.abs(np.diff(border_row))
        assert np.max(diffs) < 0.07, f"Max diff on border: {np.max(diffs):.4f}"

    def test_output_shape_and_dtype(self):
        """Форма == входной, dtype == float32, диапазон [0, 1]."""
        gray = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        mask = detail_map.detail_mask(gray)
        assert mask.shape == (100, 200)
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])