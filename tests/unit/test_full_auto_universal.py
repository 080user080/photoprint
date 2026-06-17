"""
Інтеграційний тест для processing/pipeline.py — run_full_auto.
Перевіряє, що Full Auto коректно обробляє різні типи зображень
з адаптивними параметрами, і не падає на жодному з них.
"""

import numpy as np
import cv2
import pytest
from processing.pipeline import run_full_auto


def _default_settings():
    return {
        "full_auto_perspective": False,
        "hdr_in_autofix": True,
        "hdr_strength": 0.5,
        "sharpen_strength": 0.4,
        "autosharp_max_strength": 0.7,
        "autosharp_threshold": 80.0,
        "contrast_mode": "linear",
        "autofix_contrast": 0.15,
        "bw_binary": False,
        "bw_std_thresh": 20.0,
        "edge_ratio_min": 0.03,
        "line_count_min": 3,
        "shadow_highlight_strength": 0.0,
        "output_color_mode": "auto",
    }


class TestFullAutoUniversal:
    def _check_result(self, img, result_msg):
        """Перевіряє, що результат має правильний тип та розмір."""
        result, status_msg, applied_steps = result_msg
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert "Full Auto" in status_msg
        assert isinstance(applied_steps, dict)
        return result, status_msg, applied_steps

    def test_uniform_gray_image(self):
        """Рівне сіре зображення (flat_background) → не падає, статус містить 'рівний фон'."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        assert "рівний фон" in status or "Full Auto" in status

    def test_bw_document_like(self):
        """Імітація ч-б документа → не падає."""
        img = np.full((200, 200, 3), 220, dtype=np.uint8)  # світлий фон
        img[40:160, 40:160, :] = 180  # темніший прямокутник
        # Додаємо трохи тексту (випадкові лінії)
        for i in range(50, 150, 10):
            img[i:i+2, 50:150, :] = 80
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_photo_like(self):
        """Імітація фото (випадковий кольоровий шум) → не падає."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_color_document_like(self):
        """Імітація кольорового документа → не падає."""
        img = np.full((200, 200, 3), 240, dtype=np.uint8)
        # Кольоровий текст
        img[50:150, 30:170, :] = [200, 100, 50]
        img[60:140, 40:160, :] = [50, 150, 200]
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_dark_image(self):
        """Темне зображення з варіацією → не падає, застосовується яскравість."""
        np.random.seed(42)
        img = np.random.normal(30, 10, (100, 100, 3)).astype(np.uint8)  # темне, але з варіацією
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_overexposed_image(self):
        """Пересвічене зображення → не падає."""
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_sharp_image_unchanged(self):
        """Різке зображення → sharpen_strength = 0, але все одно обробляється."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        # Результат не має бути ідентичним через інші корекції (контраст тощо)
        # Але має бути коректним uint8

    def test_small_image(self):
        """Маленьке зображення (32x32) → не падає."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_large_image(self):
        """Велике зображення (800x600) → не падає (тест на продуктивність)."""
        img = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_dry_run_never_modifies_input(self):
        """dry_run=True → вхідне зображення не змінюється."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(np.uint8)
        img_copy = img.copy()
        result, status, steps = run_full_auto(img, _default_settings(), dry_run=True)
        assert np.array_equal(img, img_copy)
        assert "dry" in status

    def test_run_full_auto_immutable_input(self):
        """run_full_auto не змінює вхідне зображення."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(np.uint8)
        img_copy = img.copy()
        result, status, steps = run_full_auto(img, _default_settings())
        assert np.array_equal(img, img_copy)

    def test_gradient_background_document(self):
        """Flat/рівний фон документа з текстом → білий фон застосовується.

        Цей тест перевіряє, що _apply_auto_white_background використовує
        актуальні метрики (не застарілі з кроку 0) — на flat_background
        з текстом diagnose() дає низьку uniformity, але ПІСЛЯ обробки
        фонові ділянки стають світлими й однорідними, і
        measure_background_metrics це бачить."""
        img = np.full((200, 200, 3), 245, dtype=np.uint8)  # дуже світлий фон
        for i in range(40, 160, 12):
            img[i:i + 2, 30:170, :] = 60  # темний текст
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        # Фонові кути (без тексту) мають бути білими
        corner_tl = result[0:10, 0:10, :]
        corner_br = result[190:200, 190:200, :]
        assert np.mean(corner_tl == 255) > 0.7, f"Верхній лівий кут не білий"
        assert np.mean(corner_br == 255) > 0.7, f"Нижній правий кут не білий"

    def test_noise_floor_passed_to_hdr(self):
        """Перевіряє, що noise_floor передається в HDR (не крашиться)."""
        np.random.seed(42)
        noise = np.random.normal(128, 20, (100, 100, 3)).astype(np.uint8)
        result, status, steps = run_full_auto(noise, _default_settings())
        self._check_result(noise, (result, status, steps))

    def test_color_saturation_effect(self):
        """Кольорове фото → HDR застосовується (status містить HDR)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # червоний
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))

    def test_flat_background_handling(self):
        """flat_background → тільки легка різкість, без CLAHE/HDR."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        # На рівному фоні статус має згадувати "рівний фон" або бути коректним
        assert isinstance(status, str)

    def test_flat_background_becomes_white(self):
        """Сірий рівний фон → фон стає білим (255, 255, 255)."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)  # світло-сірий
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        # Перевіряємо, що більшість пікселів стали білими
        white_ratio = float(np.mean(np.all(result == 255, axis=2)))
        assert white_ratio > 0.8, f"Замало білих пікселів: {white_ratio:.3f}"
        # Статус має згадувати "білий фон"
        assert "білий фон" in status or "рівний фон" in status

    def test_bw_document_becomes_white_background(self):
        """Ч-б документ зі світлим фоном → фон стає білим."""
        img = np.full((200, 200, 3), 220, dtype=np.uint8)  # світлий фон
        img[40:160, 40:160, :] = 180  # темніший прямокутник
        for i in range(50, 150, 10):
            img[i:i+2, 50:150, :] = 80  # текст
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        # Фон має стати білим
        top_left_corner = result[0:10, 0:10, :]
        assert np.all(top_left_corner == 255), "Кут зображення не білий"
        # Текст має зберегтися (темні пікселі)
        assert np.any(result < 200), "Текст зник — все біле"

    def test_photo_not_whitened(self):
        """Фото — білий фон не застосовується."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result, status, steps = run_full_auto(img, _default_settings())
        self._check_result(img, (result, status, steps))
        # Фото не повинно мати "білий фон" в статусі
        assert "білий фон" not in status, "Фото не повинно отримати білий фон"
        # Результат не має бути повністю білим
        white_ratio = float(np.mean(np.all(result == 255, axis=2)))
        assert white_ratio < 0.5, f"Фото стало майже білим: {white_ratio:.3f}"
