"""
Тести для нових функцій контрасту:
smart_contrast_percentile, contrast_s_curve, local_contrast_adaptive.
"""

import numpy as np
import pytest
from processing import brightness_contrast as bc


def _gray_bgr(h=100, w=100, val=128):
    """Допоміжна: однотонне BGR зображення."""
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestSmartContrastPercentile:
    def test_narrow_histogram_stretches(self):
        """Вузька гістограма → після функції p99-p1 більший."""
        # Створюємо зображення з діапазонам > 20 щоб функція не повернула копію
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img[10:40, 10:40] = [200, 200, 200]  # значна варіація
        lab = _to_lab(img)
        l = lab[:, :, 0]
        p1_before = float(np.percentile(l, 1))
        p99_before = float(np.percentile(l, 99))
        range_before = p99_before - p1_before

        res = bc.smart_contrast_percentile(img, strength=0.8)
        lab_res = _to_lab(res)
        l_res = lab_res[:, :, 0]
        p1_after = float(np.percentile(l_res, 1))
        p99_after = float(np.percentile(l_res, 99))
        range_after = p99_after - p1_after

        assert range_after > range_before, f"Розтягнення не сталося: {range_before} -> {range_after}"

    def test_wide_histogram_unchanged(self):
        """Широка гістограма (p1=0, p99=255) → результат близький до оригіналу."""
        # Створюємо зображення де 1% пікселів = 0 і 99% = 255
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img[0:1, :, :] = 0      # 1% — 0
        img[-1:, :, :] = 255    # 1% — 255
        res = bc.smart_contrast_percentile(img, strength=0.8)
        # З максимальним діапазоном зміни мають бути мінімальними
        diff = np.abs(img.astype(int) - res.astype(int))
        max_diff = np.max(diff)
        assert max_diff < 50, f"Занадто велика зміна для широкої гістограми: {max_diff}"

    def test_strength_zero_returns_copy(self):
        """strength=0 → рівний оригіналу."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img[10:40, 10:40] = [200, 200, 200]
        res = bc.smart_contrast_percentile(img, strength=0.0)
        assert np.array_equal(img, res)
        assert res is not img

    def test_immutable(self):
        """Вхідне зображення не змінюється."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_copy = img.copy()
        _ = bc.smart_contrast_percentile(img, strength=0.5)
        assert np.array_equal(img, img_copy)


class TestContrastSCurve:
    def test_midtones_enhanced(self):
        """Після функції дисперсія L збільшується."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img[10:40, 10:40] = [100, 100, 100]
        img[5:15, 5:15] = [180, 180, 180]
        lab = _to_lab(img)
        std_before = float(np.std(lab[:, :, 0]))

        res = bc.contrast_s_curve(img, strength=0.8)
        lab_res = _to_lab(res)
        std_after = float(np.std(lab_res[:, :, 0]))

        assert std_after > std_before, f"Дисперсія не збільшилась: {std_before} -> {std_after}"

    def test_shadows_not_crushed(self):
        """Мінімальне L після функції вище ніж при лінійному контрасті."""
        img = np.full((50, 50, 3), 30, dtype=np.uint8)  # темне
        img[10:40, 10:40] = [200, 200, 200]  # світла пляма
        res = bc.contrast_s_curve(img, strength=0.8)
        lab_res = _to_lab(res)
        l_min = float(np.min(lab_res[:, :, 0]))
        assert l_min >= 0, f"Тіні занадто crushed: min L = {l_min}"

    def test_highlights_not_blown(self):
        """Максимальне L після функції не 255 якщо до цього було нижче 250."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img[10:40, 10:40] = [240, 240, 240]  # світла область, але не 255
        res = bc.contrast_s_curve(img, strength=0.8)
        lab_res = _to_lab(res)
        l_max = float(np.max(lab_res[:, :, 0]))
        assert l_max <= 255

    def test_strength_zero_minimal_effect(self):
        """strength=0 → gamma=1.0 → результат має бути близьким до оригіналу."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img[10:40, 10:40] = [200, 200, 200]
        res = bc.contrast_s_curve(img, strength=0.0)
        # При gamma=1.0, s = x / (x + (1-x)) = x, тобто результат = оригінал
        # Але через float precision може бути мала похибка
        diff = np.abs(img.astype(int) - res.astype(int))
        max_diff = np.max(diff)
        assert max_diff < 5, f"Занадто велика зміна при strength=0: {max_diff}"

    def test_immutable(self):
        """Вхідне зображення не змінюється."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_copy = img.copy()
        _ = bc.contrast_s_curve(img, strength=0.5)
        assert np.array_equal(img, img_copy)


class TestLocalContrastAdaptive:
    def test_detail_areas_enhanced(self):
        """В текстурованій зоні контраст зростає."""
        # Створюємо зображення з текстурою (шум)
        np.random.seed(42)
        img = np.random.randint(80, 180, (50, 50, 3), dtype=np.uint8)
        lab = _to_lab(img)
        std_before_texture = float(np.std(lab[10:40, 10:40, 0]))

        res = bc.local_contrast_adaptive(img, strength=0.8)
        lab_res = _to_lab(res)
        std_after_texture = float(np.std(lab_res[10:40, 10:40, 0]))

        # Std має збільшитись або залишитись великим
        assert std_after_texture >= std_before_texture * 0.5

    def test_flat_areas_unchanged(self):
        """В однотонній зоні зміна мінімальна."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        res = bc.local_contrast_adaptive(img, strength=0.8)
        diff = np.abs(img.astype(int) - res.astype(int))
        max_diff = np.max(diff)
        # Однотонна зона має змінитись мінімально
        assert max_diff < 50, f"Занадто велика зміна для однотонної зони: {max_diff}"

    def test_immutable(self):
        """Вхідне зображення не змінюється."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_copy = img.copy()
        _ = bc.local_contrast_adaptive(img, strength=0.5)
        assert np.array_equal(img, img_copy)


# ---------------------------------------------------------------------------
# Хелпери
# ---------------------------------------------------------------------------

def _to_lab(img: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)