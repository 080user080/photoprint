"""
Unit-тести для doc_classifier.py: classify() та _has_color_content().
"""

import numpy as np
import cv2
import pytest
from processing.doc_classifier import (
    classify, _has_color_content, _has_histogram_color_content,
    _is_flat_background, CHROMA_PIXEL_THRESHOLD, COLOR_PIXEL_RATIO_MIN,
    FLAT_BG_UNIFORMITY_THRESH, FLAT_BG_DETAIL_DENSITY_THRESH,
    FLAT_BG_EDGE_RATIO_MAX, FLAT_BG_LINE_COUNT_MAX,
)


# ============================================================
# Тест 1: _has_color_content — перевірка допоміжної функції
# ============================================================

class TestHasColorContent:
    """Unit-тести для _has_color_content на контрольованих LAB-масивах."""

    def test_all_neutral(self):
        """a=b=128 скрізь — has_color_content=False."""
        h, w = 100, 100
        a_ch = np.full((h, w), 128, dtype=np.uint8)
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        ratio, has = _has_color_content(a_ch, b_ch)
        assert ratio == 0.0
        assert has is False

    def test_all_pale_red(self):
        """a злегка вище 128 (блідо-червоний) — |140-128|=12 >= CHROMA_PIXEL_THRESHOLD=10."""
        h, w = 100, 100
        a_ch = np.full((h, w), 140, dtype=np.uint8)  # |140-128|=12 >= CHROMA_PIXEL_THRESHOLD=10
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        ratio, has = _has_color_content(a_ch, b_ch)
        assert ratio == 1.0
        assert has is True

    def test_all_neutral_light(self):
        """a з невеликим відхиленням (нижче порогу) — has_color_content=False."""
        h, w = 100, 100
        a_ch = np.full((h, w), 135, dtype=np.uint8)  # |135-128|=7 < CHROMA_PIXEL_THRESHOLD=10
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        ratio, has = _has_color_content(a_ch, b_ch)
        assert ratio == 0.0
        assert has is False

    def test_mixed_2pct_color(self):
        """2% кольорових пікселів (> COLOR_PIXEL_RATIO_MIN=0.01) — has_color_content=True."""
        h, w = 100, 100
        a_ch = np.full((h, w), 128, dtype=np.uint8)
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        # Фарбуємо 2% пікселів у синій (b >> 128)
        n_color = int(h * w * 0.02)
        a_ch.flat[:n_color] = 128
        b_ch.flat[:n_color] = 180  # |180-128|=52 >= CHROMA_PIXEL_THRESHOLD=10
        ratio, has = _has_color_content(a_ch, b_ch)
        assert ratio >= 0.02
        assert has is True

    def test_mixed_0_5pct_color(self):
        """0.5% кольорових пікселів (< COLOR_PIXEL_RATIO_MIN=0.01) — has_color_content=False."""
        h, w = 100, 100
        a_ch = np.full((h, w), 128, dtype=np.uint8)
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        n_color = int(h * w * 0.005)
        a_ch.flat[:n_color] = 128
        b_ch.flat[:n_color] = 180
        ratio, has = _has_color_content(a_ch, b_ch)
        assert ratio < COLOR_PIXEL_RATIO_MIN
        assert has is False

    def test_dtype_preserved(self):
        """Вхідні uint8, повертає tuple[float, bool]."""
        h, w = 50, 50
        a_ch = np.full((h, w), 128, dtype=np.uint8)
        b_ch = np.full((h, w), 128, dtype=np.uint8)
        ratio, has = _has_color_content(a_ch, b_ch)
        assert isinstance(ratio, float)
        assert isinstance(has, bool)


# ============================================================
# Тест 2: color subject on white background ≠ bw_document
# ============================================================

class TestColorSubjectOnWhiteBg:
    """
    Велике біле/світло-сіре поле (>80% кадру) + невеликий
    насичено-кольоровий прямокутник + лінії (рамка).
    classify() не має повернути "bw_document".
    """

    def test_not_bw_document(self):
        h, w = 300, 300
        # Білий фон
        bgr = np.full((h, w, 3), 240, dtype=np.uint8)

        # Кольоровий прямокутник 60x60 (~4% кадру) (червоний) у центрі
        cy, cx = h // 2, w // 2
        half = 30
        bgr[cy - half:cy + half, cx - half:cx + half] = (0, 0, 200)  # BGR: червоний

        # Лінії (рамка) — достатньо для edge_ratio/line_count
        cv2.rectangle(bgr, (10, 10), (w - 10, h - 10), (0, 0, 0), 2)

        result = classify(bgr)

        assert result != "bw_document", \
            f"Кольорове фото з білим фоном класифіковано як {result}"


# ============================================================
# Тест 3: true grayscale scan — все ще bw_document
# ============================================================

class TestTrueGrayscaleScan:
    """
    Синтетичний скан без кольору (a=b=128 скрізь, ±шум) + текстові лінії.
    classify() має повернути "bw_document" (regression).
    """

    def test_still_bw_document(self):
        h, w = 400, 400
        # Сірий фон
        l_ch = np.full((h, w), 200, dtype=np.uint8)
        a_ch = np.full((h, w), 128, dtype=np.uint8)
        b_ch = np.full((h, w), 128, dtype=np.uint8)

        # Додаємо невеликий шум для реалістичності
        rng = np.random.default_rng(42)
        l_ch = np.clip(l_ch.astype(np.float32) + rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)

        # Текстові/лінійні елементи (товсті лінії, що детектяться HoughLinesP)
        cv2.line(l_ch, (20, 100), (380, 100), (0,), 3)  # горизонтальна лінія
        cv2.line(l_ch, (20, 200), (380, 200), (0,), 3)
        cv2.line(l_ch, (20, 300), (380, 300), (0,), 3)

        lab = cv2.merge([l_ch, a_ch, b_ch])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        result = classify(bgr)
        assert result == "bw_document", \
            f"Справжній ч-б скан класифіковано як {result} (очікувалось bw_document)"


# ============================================================
# Тест 4: low-saturation photo with lines — не bw_document
# ============================================================

class TestLowSaturationPhoto:
    """
    Фото з низькою, але присутньою насиченістю по всьому кадру
    (рівномірно невеликий a/b deviation, ±10) і з лініями.
    Не має стати "bw_document".
    """

    def test_not_bw_document(self):
        h, w = 200, 200
        # Фон з невеликою насиченістю (a = 128 ± 10, b = 128 ± 10)
        rng = np.random.default_rng(123)
        a_ch = np.clip(128 + rng.normal(0, 10, (h, w)).astype(np.int16), 0, 255).astype(np.uint8)
        b_ch = np.clip(128 + rng.normal(0, 10, (h, w)).astype(np.int16), 0, 255).astype(np.uint8)

        # L-канал з варіацією
        l_ch = np.clip(128 + rng.normal(0, 30, (h, w)).astype(np.int16), 0, 255).astype(np.uint8)

        # Лінії (рамка)
        cv2.rectangle(l_ch, (5, 5), (w - 5, h - 5), (0,), 2)

        lab = cv2.merge([l_ch, a_ch, b_ch])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        result = classify(bgr)
        # Низька насиченість ±10 може давати std_a/std_b < 10 (нового порогу),
        # але наявність пікселів з |a-128| або |b-128| > CHROMA_PIXEL_THRESHOLD=10
        # має запобігти bw_document через has_color_content або has_histogram_color
        assert result != "bw_document", \
            f"Фото з низькою насиченістю класифіковано як {result}"


# ============================================================
# Тест 5: _is_flat_background — перевірка рівного фону (TODO 4.3)
# ============================================================

class TestIsFlatBackground:
    """Unit-тести для _is_flat_background з новими перевірками (Canny, HoughLinesP)."""

    def test_uniform_gray_is_flat(self):
        """Однотонне сіре зображення — flat_background=True."""
        h, w = 200, 200
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        assert _is_flat_background(img) is True

    def test_uniform_white_is_flat(self):
        """Однотонне біле зображення — flat_background=True."""
        h, w = 200, 200
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        assert _is_flat_background(img) is True

    def test_uniform_black_is_flat(self):
        """Однотонне чорне зображення — flat_background=True."""
        h, w = 200, 200
        img = np.full((h, w, 3), 0, dtype=np.uint8)
        assert _is_flat_background(img) is True

    def test_scan_with_lines_not_flat(self):
        """
        Скан з текстом/лініями на рівному фоні — flat_background=False.
        Навіть при високій uniformity, наявність ліній (HoughLinesP)
        має запобігти класифікації як flat_background.
        """
        h, w = 300, 300
        img = np.full((h, w, 3), 240, dtype=np.uint8)

        # Товсті лінії (текст/рамка) — детектуються HoughLinesP
        cv2.line(img, (20, 50), (280, 50), (0, 0, 0), 3)
        cv2.line(img, (20, 150), (280, 150), (0, 0, 0), 3)
        cv2.line(img, (20, 250), (280, 250), (0, 0, 0), 3)

        assert _is_flat_background(img) is False, \
            "Скан з лініями не має бути flat_background"

    def test_scan_with_edges_not_flat(self):
        """
        Зображення з краями (Canny) — flat_background=False.
        Навіть при високій uniformity, наявність країв має запобігти
        класифікації як flat_background.
        """
        h, w = 300, 300
        img = np.full((h, w, 3), 240, dtype=np.uint8)

        # Прямокутник (рамка документа) — створює краї
        cv2.rectangle(img, (30, 30), (w - 30, h - 30), (0, 0, 0), 2)

        assert _is_flat_background(img) is False, \
            "Зображення з рамкою не має бути flat_background"

    def test_scan_with_text_not_flat(self):
        """
        Скан з текстом (багато дрібних деталей) — flat_background=False.
        Текст створює краї та лінії, які мають запобігти flat_background.
        """
        h, w = 300, 300
        img = np.full((h, w, 3), 240, dtype=np.uint8)

        # Імітуємо текст: багато прямокутників/ліній достатнього розміру,
        # щоб їх виявив Canny/HoughLinesP після ресайзу до 0.3
        rng = np.random.default_rng(7)
        for _ in range(50):
            x = int(rng.integers(10, w - 60))
            y = int(rng.integers(10, h - 40))
            cv2.rectangle(img, (x, y), (x + 40, y + 20), (0, 0, 0), 2)

        assert _is_flat_background(img) is False, \
            "Скан з текстом не має бути flat_background"

    def test_classify_scan_not_flat_background(self):
        """
        Regression: скан з текстом/лініями не має класифікуватись як
        flat_background (TODO 4.3 — flat_background поглинає скани).
        """
        h, w = 300, 300
        img = np.full((h, w, 3), 240, dtype=np.uint8)

        # Товсті лінії (текст/рамка)
        cv2.line(img, (20, 50), (280, 50), (0, 0, 0), 3)
        cv2.line(img, (20, 150), (280, 150), (0, 0, 0), 3)
        cv2.line(img, (20, 250), (280, 250), (0, 0, 0), 3)

        result = classify(img)
        assert result != "flat_background", \
            f"Скан з лініями класифіковано як {result} (очікувалось не flat_background)"

    def test_classify_uniform_gray_is_flat_background(self):
        """Однотонне сіре зображення — classify() повертає flat_background."""
        h, w = 200, 200
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        result = classify(img)
        assert result == "flat_background", \
            f"Однотонне сіре зображення класифіковано як {result} (очікувалось flat_background)"
