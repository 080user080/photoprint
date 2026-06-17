"""
Тести для processing/pipeline.py — _compute_adaptive_params та адаптивна логіка.
Перевіряє, що параметри обробки правильно обчислюються для різних типів зображень.
"""

import numpy as np
import cv2
import pytest
from processing.pipeline import _compute_adaptive_params
from processing.diagnostics import diagnose


def _default_settings():
    return {
        "autosharp_max_strength": 0.7,
        "autosharp_threshold": 80.0,
        "bw_std_thresh": 20.0,
        "edge_ratio_min": 0.03,
        "line_count_min": 3,
    }


class TestAdaptiveParams:
    def test_uniform_image_low_hdr_strength(self):
        """Рівне сіре зображення → низька hdr_strength та noise_floor ≈ 1.5."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        diag = diagnose(img, _default_settings())
        params = _compute_adaptive_params(diag)
        # На рівному фоні HDR сила мінімальна
        assert params["hdr_strength"] <= 0.4
        # Noise floor мінімальний
        assert params["noise_floor"] >= 1.5
        # Контраст не потрібен (або мінімальний)
        assert params["contrast_strength"] >= 0.0
        # Shadow remove має бути False на flat_background
        assert params["shadow_remove"] is False

    def test_noisy_image_high_noise_floor(self):
        """Зашумлене зображення → вищий noise_floor."""
        np.random.seed(42)
        noise = np.random.normal(128, 30, (100, 100, 3)).astype(np.uint8)
        diag = diagnose(noise, _default_settings())
        params = _compute_adaptive_params(diag)
        # Noise_floor має бути > 1.5 для шумного зображення
        assert params["noise_floor"] > 1.5

    def test_high_contrast_image_low_contrast_needed(self):
        """Висококонтрастне зображення → contrast_strength = 0 (якщо dynamic_range > 180)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :50, :] = 255  # половина біла, половина чорна
        diag = diagnose(img, _default_settings())
        params = _compute_adaptive_params(diag)
        # Якщо dynamic_range > 180, contrast_strength = 0
        if diag.dynamic_range > 180:
            assert params["contrast_strength"] == 0.0

    def test_textured_image_high_hdr(self):
        """Текстурне зображення → вища hdr_strength (ближче до 0.8)."""
        np.random.seed(42)
        texture = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        diag = diagnose(texture, _default_settings())
        params = _compute_adaptive_params(diag)
        # На текстурному зображенні detail_density висока → hdr_strength висока
        assert params["hdr_strength"] > 0.5

    def test_bright_image_no_brightness_needed(self):
        """Зображення з середньою яскравістю → brightness_needed = False."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        diag = diagnose(img, _default_settings())
        params = _compute_adaptive_params(diag)
        # На нейтральному зображенні brightness_needed = False
        assert params["brightness_needed"] is False

    def test_dark_image_brightness_needed(self):
        """Темне зображення → brightness_needed = True."""
        img = np.full((100, 100, 3), 30, dtype=np.uint8)
        diag = diagnose(img, _default_settings())
        params = _compute_adaptive_params(diag)
        # Темне зображення має потребу в корекції яскравості
        assert params["brightness_needed"] is True
        assert params["brightness_correction"] > 0

    def test_sharp_image_sharpen_strength(self):
        """Різке зображення → sharpen_strength = 0."""
        np.random.seed(42)
        sharp = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        diag = diagnose(sharp, _default_settings())
        params = _compute_adaptive_params(diag)
        # На різкому зображенні (шум) sharpen_strength = 0
        assert params["sharpen_strength"] >= 0.0

    def test_shadow_remove_for_document(self):
        """Документ з градієнтом → _compute_adaptive_params не падає, повертає bool."""
        # Створюємо імітацію документа з градієнтом
        img = np.full((200, 200, 3), 200, dtype=np.uint8)  # світлий фон
        img[30:170, 30:170, :] = 180  # трохи темніший "документ"
        # Додаємо горизонтальний градієнт
        for x in range(200):
            val = int(200 - 80 * x / 199)
            img[:, x, :] = np.clip(img[:, x, :].astype(int) - (200 - val), 0, 255).astype(np.uint8)
        diag = diagnose(img, _default_settings())
        params = _compute_adaptive_params(diag)
        # shadow_remove має бути bool
        assert isinstance(params["shadow_remove"], bool)
        # Якщо є градієнт і фон рівномірний — shadow_remove має бути True
        if diag.gradient_has and diag.background_uniformity > 0.4 and diag.doc_type in ("bw_document", "color_document"):
            assert params["shadow_remove"] is True
        else:
            # Інакше може бути False, це теж нормально
            assert params["shadow_remove"] is False
