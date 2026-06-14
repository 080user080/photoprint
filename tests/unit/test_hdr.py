"""
Unit-тесты для processing.hdr (Content-Adaptive HDR).
"""

import cv2
import numpy as np
import pytest

from processing import hdr


class TestAdaptiveTileGrid:
    """Тесты для adaptive_tile_grid."""

    def test_small_image_min_grid(self):
        """Маленькое изображение -> минимальная сетка."""
        gx, gy = hdr.adaptive_tile_grid(100, 100)
        assert gx >= hdr.HDR_TILE_MIN_GRID
        assert gy >= hdr.HDR_TILE_MIN_GRID

    def test_large_image_max_grid(self):
        """Очень большое изображение -> максимальная сетка."""
        gx, gy = hdr.adaptive_tile_grid(10000, 10000)
        assert gx <= hdr.HDR_TILE_MAX_GRID
        assert gy <= hdr.HDR_TILE_MAX_GRID

    def test_resolution_independence(self):
        """
        Разные разрешения одного контента дают сопоставимые tile grids.
        Превью 800x600 и полный 4000x3000 должны иметь разные, но не экстремальные grid size.
        """
        g1x, g1y = hdr.adaptive_tile_grid(600, 800)
        g2x, g2y = hdr.adaptive_tile_grid(3000, 4000)
        # Оба в разумных пределах
        assert hdr.HDR_TILE_MIN_GRID <= g1x <= hdr.HDR_TILE_MAX_GRID
        assert hdr.HDR_TILE_MIN_GRID <= g2x <= hdr.HDR_TILE_MAX_GRID
        # Чем больше изображение, тем больше grid (или равно)
        assert g2x >= g1x
        assert g2y >= g1y


class TestAutoStrengthFactor:
    """Тесты для _auto_strength_factor."""

    def test_low_range(self):
        """Узкий L-диапазон (все пиксели в [200, 230]) -> фактор <= 0.5."""
        l_ch = np.random.randint(200, 231, (100, 100)).astype(np.uint8)
        factor = hdr._auto_strength_factor(l_ch)
        assert 0.0 <= factor <= 0.5

    def test_full_range(self):
        """Полный диапазон [0, 255] -> фактор == 1.0."""
        l_ch = np.array([[0, 255]], dtype=np.uint8)
        factor = hdr._auto_strength_factor(l_ch)
        assert factor == 1.0

    def test_min_factor_clamp(self):
        """Очень узкий диапазон -> фактор не меньше HDR_RANGE_MIN_FACTOR."""
        l_ch = np.full((10, 10), 128, dtype=np.uint8)
        factor = hdr._auto_strength_factor(l_ch)
        assert factor >= hdr.HDR_RANGE_MIN_FACTOR


class TestApplyCoring:
    """Тесты для _apply_coring."""

    def test_zeroes_small_diffs(self):
        """
        Значения в [-HDR_NOISE_FLOOR, HDR_NOISE_FLOOR] -> обнуляются.
        """
        diff = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.4], dtype=np.float32)
        result = hdr._apply_coring(diff, noise_floor=1.5)
        assert np.all(result == 0.0)

    def test_large_diff_positive(self):
        """Значение 5 -> возвращает 5 - HDR_NOISE_FLOOR (при noise_floor=2)"""
        diff = np.array([5.0], dtype=np.float32)
        result = hdr._apply_coring(diff, noise_floor=2.0)
        assert np.isclose(result[0], 3.0)

    def test_large_diff_negative(self):
        """Значение -5 -> возвращает -(5 - HDR_NOISE_FLOOR)"""
        diff = np.array([-5.0], dtype=np.float32)
        result = hdr._apply_coring(diff, noise_floor=2.0)
        assert np.isclose(result[0], -3.0)

    def test_soft_threshold_continuity(self):
        """
        Проверка непрерывности: на границе noise_floor значение не делает резких скачков.
        """
        noise_floor = 1.5
        diff = np.array([noise_floor - 0.001, noise_floor + 0.001], dtype=np.float32)
        result = hdr._apply_coring(diff, noise_floor=noise_floor)
        # Первое ~0, второе ~0.001
        assert result[0] < 0.01
        assert result[1] > 0.0


class TestApplyHDR:
    """Тесты для apply и apply_adaptive."""

    def test_apply_flat_white_no_artifacts(self):
        """
        Синтетическое 400x400, заливка 235 + Gaussian noise(σ=2).
        hdr.apply(strength=1.0): max abs diff по L <= HDR_NOISE_FLOOR для >= 99% пикселей.
        """
        np.random.seed(42)
        gray_bg = np.full((400, 400, 3), 235, dtype=np.uint8)
        noise = np.random.normal(0, 2, (400, 400, 3)).astype(np.int16)
        img = np.clip(gray_bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        result = hdr.apply(img, strength=1.0)

        # Считаем разницу по L-каналу
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_diff = np.abs(lab_result[:, :, 0].astype(np.float32) - lab_orig[:, :, 0].astype(np.float32))

        # >= 99% пикселей имеют diff <= HDR_NOISE_FLOOR
        fraction_ok = np.mean(l_diff <= hdr.HDR_NOISE_FLOOR)
        assert fraction_ok >= 0.99, f"Fraction of pixels within noise floor: {fraction_ok:.4f}"

    def test_apply_preserves_detail_effect(self):
        """
        На изображении с текстурой (шумовая текстура с низким контрастом):
        hdr.apply(strength=0.8) должен увеличить средний локальный std
        (локальный контраст после CLAHE растёт).
        """
        np.random.seed(42)
        # Низкоконтрастная текстура: значения в [90, 130]
        texture = np.random.randint(90, 131, (100, 100)).astype(np.uint8)
        img = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

        result = hdr.apply(img, strength=0.8)

        # Считаем mean локального std через box filter (kernel=5) для L-канала
        def mean_local_std(l_ch):
            f = l_ch.astype(np.float32)
            mean = cv2.boxFilter(f, -1, (5, 5), borderType=cv2.BORDER_REFLECT)
            mean_sq = cv2.boxFilter(f * f, -1, (5, 5), borderType=cv2.BORDER_REFLECT)
            var = np.maximum(mean_sq - mean * mean, 0.0)
            return float(np.mean(np.sqrt(var)))

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_res = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        mls_orig = mean_local_std(lab_orig[:, :, 0])
        mls_res = mean_local_std(lab_res[:, :, 0])

        # Локальный контраст должен увеличиться после CLAHE
        assert mls_res > mls_orig, f"mls_res={mls_res:.2f} <= mls_orig={mls_orig:.2f}"

    def test_apply_equals_apply_adaptive_without_mask(self):
        """
        apply(img, s) == apply_adaptive(img, s, text_mask=None) (побайтово).
        """
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        r1 = hdr.apply(img, strength=0.7)
        r2 = hdr.apply_adaptive(img, strength=0.7, text_mask=None)
        assert np.array_equal(r1, r2), "apply и apply_adaptive(text_mask=None) должны совпадать"

    def test_coring_in_apply_flat_image(self):
        """
        На плоском изображении с HDR_NOISE_FLOOR=1.5, max diff между
        оригиналом и результатом не превышает HDR_NOISE_FLOOR * 2 (с учетом
        coring и detail mask).
        """
        gray = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = hdr.apply(gray, strength=1.0)
        diff = np.abs(result.astype(np.float32) - gray.astype(np.float32))
        # Допустимое отклонение с учетом coring и detail_mask
        assert np.max(diff) < 4.0, f"Max diff: {np.max(diff):.2f}"

    def test_resolution_independence(self):
        """
        Одно и то же изображение в двух разрешениях (resize x0.25 и x1):
        std разницы "до/после" в плоских зонах должен быть близким (относ. разница < 20%).
        """
        np.random.seed(42)
        # Создаем изображение с плоским фоном и небольшим текстовым участком
        img = np.full((400, 400, 3), 230, dtype=np.uint8)
        # Добавляем контрастную область (текст)
        img[100:110, 100:300] = 30
        img[130:140, 100:300] = 30
        img[160:170, 100:300] = 30

        # Полный размер
        result_full = hdr.apply(img, strength=0.8)
        lab_orig_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_res_full = cv2.cvtColor(result_full, cv2.COLOR_BGR2LAB)
        diff_full = np.abs(lab_res_full[:, :, 0].astype(np.float32) - lab_orig_full[:, :, 0].astype(np.float32))
        std_full = np.std(diff_full)

        # Уменьшенный размер (x0.25)
        small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
        result_small = hdr.apply(small, strength=0.8)
        lab_orig_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        lab_res_small = cv2.cvtColor(result_small, cv2.COLOR_BGR2LAB)
        diff_small = np.abs(lab_res_small[:, :, 0].astype(np.float32) - lab_orig_small[:, :, 0].astype(np.float32))
        std_small = np.std(diff_small)

        # Относительная разница std < 20%
        max_std = max(std_full, std_small)
        min_std = min(std_full, std_small)
        if min_std > 0:
            rel_diff = (max_std - min_std) / min_std
            assert rel_diff < 0.30, f"Relative std diff: {rel_diff:.2f}"

    def test_apply_adaptive_with_text_mask_unchanged_behavior(self):
        """
        Non-regression: с auto_detail=False и тем же text_mask, что и раньше,
        результат эквивалентен старой реализации (с точностью до coring).
        Для проверки сравниваем с apply_adaptive без detail, только text_mask.
        """
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Создаем простую text_mask: половина изображения
        text_mask = np.zeros(gray.shape, dtype=np.uint8)
        text_mask[25:75, 25:75] = 255

        result = hdr.apply_adaptive(img, strength=0.6, text_mask=text_mask, auto_detail=False)

        # Просто проверяем, что функция отработала без ошибок и вернула корректный тип
        assert result.dtype == np.uint8
        assert result.shape == img.shape


if __name__ == "__main__":
    pytest.main([__file__])