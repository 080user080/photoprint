"""
Unit-тести для autofix.py: _step_lab_clahe_normalize.
"""

import numpy as np
import cv2
import pytest
from processing import autofix


# -- допоміжні функції --

def _make_flat_lab(h: int, w: int, l_val: float = 128.0, noise_sigma: float = 0.0) -> np.ndarray:
    """
    Створити BGR-зображення (h, w) з однотонним L≈l_val та опціональним шумом.
    """
    l_ch = np.full((h, w), l_val, dtype=np.uint8)
    if noise_sigma > 0.0:
        noise = np.random.normal(0, noise_sigma, (h, w)).astype(np.float32)
        l_ch = np.clip(l_ch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    a_ch = np.full((h, w), 128, dtype=np.uint8)
    b_ch = np.full((h, w), 128, dtype=np.uint8)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _make_rect_on_flat(h: int, w: int,
                       flat_l: float = 200.0,
                       rect_l: float = 80.0,
                       rect_size: int = 30) -> np.ndarray:
    """
    BGR-зображення з плоским фоном flat_l і контрастним квадратом rect_l
    у центрі.
    """
    l_ch = np.full((h, w), flat_l, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    half = rect_size // 2
    l_ch[cy - half:cy + half, cx - half:cx + half] = rect_l
    a_ch = np.full((h, w), 128, dtype=np.uint8)
    b_ch = np.full((h, w), 128, dtype=np.uint8)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# Тест 1: flat photo — має залишатись майже без змін
# ============================================================

class TestClaheNormalizeFlatPhoto:
    """
    _step_lab_clahe_normalize(aggressive=False) на однотонному
    зображенні L≈220 + weak Gaussian noise: макс зміна по L
    не має перевищувати noise_floor для ≥99% пікселів.
    """

    def test_flat_photo_unchanged(self):
        h, w = 220, 220
        # Заливка L=220 + шум σ=2 — типовий світлий фон
        img = _make_flat_lab(h, w, l_val=220.0, noise_sigma=2.0)

        result = autofix._step_lab_clahe_normalize(img, aggressive=False)

        # Витягуємо L-канали
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_res = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_orig = lab_orig[:, :, 0].astype(np.float32)
        l_res = lab_res[:, :, 0].astype(np.float32)

        abs_diff = np.abs(l_res - l_orig)

        # На плоскому зображенні з шумом σ=2, detail_mask може
        # трохи пропускати шумові флуктуації, а coring обнуляє
        # diff < 1.5. Залишкова зміна на окремих пікселях може
        # досягати ~5-8 рівнів. Головне — зміна не має бути
        # великою (стара реалізація давала >30 на плоскому фоні).
        max_diff = float(np.max(abs_diff))
        assert max_diff <= 15.0, \
            f"Максимальна зміна L-каналу на плоскому фоні: {max_diff:.2f} (має бути ≤15)"

        # ≥99% пікселів мають diff ≤ 10 (залишковий вплив detail_mask
        # на окремих пікселях з підвищеним локальним std через шум)
        p99 = float(np.percentile(abs_diff, 99))
        assert p99 <= 10.0, \
            f"99-й перцентиль diff на плоскому фоні: {p99:.2f} (має бути ≤10.0)"


# ============================================================
# Тест 2: detail preserved — в контрастному прямокутнику ефект є,
#         на фоні — немає
# ============================================================

class TestClaheNormalizeDetailPreserved:
    """
    Зображення з контрастним прямокутником на плоскому фоні.
    Всередині прямокутника variance L має зрости (CLAHE-ефект),
    поза ним — залишитись без змін (detail_mask ≈ 0).
    """

    def test_detail_preserved(self):
        h, w = 200, 200
        flat_l = 200.0
        rect_l = 60.0
        rect_size = 40
        img = _make_rect_on_flat(h, w, flat_l=flat_l, rect_l=rect_l,
                                 rect_size=rect_size)

        result = autofix._step_lab_clahe_normalize(img, aggressive=False)

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_res = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_orig = lab_orig[:, :, 0].astype(np.float32)
        l_res = lab_res[:, :, 0].astype(np.float32)

        cy, cx = h // 2, w // 2
        half = rect_size // 2
        inside_mask = np.zeros((h, w), dtype=bool)
        inside_mask[cy - half:cy + half, cx - half:cx + half] = True
        outside_mask = ~inside_mask

        # Варіанса всередині прямокутника має зрости
        var_inside_orig = float(np.var(l_orig[inside_mask]))
        var_inside_res = float(np.var(l_res[inside_mask]))
        assert var_inside_res > var_inside_orig * 1.1, \
            f"Variance всередині прямокутника не зросла: orig={var_inside_orig:.1f} res={var_inside_res:.1f}"

        # На фоні variance майже не змінюється
        var_outside_orig = float(np.var(l_orig[outside_mask]))
        var_outside_res = float(np.var(l_res[outside_mask]))
        # Допускаємо невелику зміну через залишковий detail_mask
        assert var_outside_res <= var_outside_orig * 2.0 + 2.0, \
            f"Variance на фоні зросла занадто: orig={var_outside_orig:.1f} res={var_outside_res:.1f}"


# ============================================================
# Тест 3: aggressive=True — статистична еквівалентність старій логіці
# ============================================================

class TestClaheNormalizeAggressive:
    """
    aggressive=True має давати статистично подібний результат
    до старої реалізації (просте CLAHE + global normalize).
    Відмінності можуть бути лише через зміну tile grid
    (з 8×8 на adaptive).
    """

    def _old_implementation(self, image: np.ndarray) -> np.ndarray:
        """Стара реалізація _step_lab_clahe_normalize для aggressive=True."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # Старий фіксований tile grid 8×8
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_ch)
        l_norm = cv2.normalize(l_clahe, None, 0, 255, cv2.NORM_MINMAX)

        merged = cv2.merge([l_norm, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def test_aggressive_statistics_similar(self):
        """
        Статистики (середнє, std, гістограма) мають бути близькими.
        """
        h, w = 256, 256
        rng = np.random.default_rng(42)
        l_ch = rng.integers(30, 230, (h, w), dtype=np.uint8)
        a_ch = np.full((h, w), 128, dtype=np.uint8)
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        old_result = self._old_implementation(img)
        new_result = autofix._step_lab_clahe_normalize(img, aggressive=True)

        lab_old = cv2.cvtColor(old_result, cv2.COLOR_BGR2LAB)
        lab_new = cv2.cvtColor(new_result, cv2.COLOR_BGR2LAB)
        l_old = lab_old[:, :, 0].astype(np.float32)
        l_new = lab_new[:, :, 0].astype(np.float32)

        # Порівнюємо статистики (допускаємо розбіжності через tile grid)
        mean_diff = abs(float(np.mean(l_old)) - float(np.mean(l_new)))
        std_diff = abs(float(np.std(l_old)) - float(np.std(l_new)))

        assert mean_diff <= 15.0, \
            f"Різниця середніх L агресивної гілки: {mean_diff:.2f}"
        assert std_diff <= 10.0, \
            f"Різниця std L агресивної гілки: {std_diff:.2f}"

    def test_aggressive_output_range(self):
        """Результат aggressive=True — uint8 [0, 255], 3-канальний BGR."""
        h, w = 128, 128
        img = _make_flat_lab(h, w, l_val=100.0)

        result = autofix._step_lab_clahe_normalize(img, aggressive=True)

        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.min() >= 0
        assert result.max() <= 255