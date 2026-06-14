"""
Тест immutability — перевіряє, що жодна функція в `processing/*` не модифікує вхідний масив.
Усі функції повинні повертати новий масив, не змінюючи image.
"""

import numpy as np
import pytest
from processing import (
    autofix,
    brightness_contrast as bc,
    hdr,
    sharpen,
    shadow_remove,
    shadow_highlight,
    perspective,
)


# ---------------------------------------------------------------------------
# Фікстура: тестове зображення (BGR uint8, ~600x800)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_image() -> np.ndarray:
    """Створює синтетичне BGR зображення 600x800 з градієнтом і текстурою."""
    h, w = 600, 800
    # Градієнт фону
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    gray_bg = ((xv + yv) / 2).astype(np.uint8)

    # Темна пляма (тінь) у центрі
    cx, cy = w // 2, h // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) < (150 ** 2)
    gray_bg[mask] = np.clip(gray_bg[mask].astype(np.int16) - 60, 0, 255).astype(np.uint8)

    # Робимо 3-канальним BGR
    image = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)

    # Додаємо кольоровий шум для тестування кольорових функцій
    color_noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, color_noise)

    return image


# Потрібен cv2 для фікстури
import cv2


# ---------------------------------------------------------------------------
# Хелпер: перевіряє, що функція не змінює вхідний масив
# ---------------------------------------------------------------------------

def _check_immutable(fn, image: np.ndarray, **kwargs) -> None:
    """Викликає fn(image, **kwargs) і перевіряє, що image не змінено."""
    before = image.copy()
    result = fn(image, **kwargs)
    assert np.array_equal(image, before), (
        f"{fn.__module__}.{fn.__name__} змінив вхідне зображення!\n"
        f"  Сума abs різниці: {int(np.sum(np.abs(image.astype(np.int16) - before.astype(np.int16))))}"
    )
    # Також перевіряємо, що результат — новий масив (не той самий об'єкт)
    assert result is not image, (
        f"{fn.__module__}.{fn.__name__} повернув той самий об'єкт, а не копію!"
    )


# ---------------------------------------------------------------------------
# Тести для autofix
# ---------------------------------------------------------------------------

class TestAutofixImmutability:
    def test_apply(self, test_image):
        _check_immutable(autofix.apply, test_image, sharpen_strength=0.4, hdr_strength=0.5, use_hdr=True)

    def test_apply_bw_document(self, test_image):
        _check_immutable(autofix.apply_bw_document, test_image, sharpen_strength=0.3, binary=False)

    def test_apply_bw_document_binary(self, test_image):
        _check_immutable(autofix.apply_bw_document, test_image, sharpen_strength=0.3, binary=True)

    def test_apply_color_document(self, test_image):
        _check_immutable(autofix.apply_color_document, test_image, sharpen_strength=0.2)


# ---------------------------------------------------------------------------
# Тести для brightness_contrast
# ---------------------------------------------------------------------------

class TestBrightnessContrastImmutability:
    def test_apply_brightness(self, test_image):
        _check_immutable(bc.apply_brightness, test_image, value=0.5)

    def test_apply_brightness_zero(self, test_image):
        _check_immutable(bc.apply_brightness, test_image, value=0.0)

    def test_auto_brightness(self, test_image):
        _check_immutable(bc.auto_brightness, test_image, percentile_low=5.0, percentile_high=95.0)

    def test_apply_contrast(self, test_image):
        _check_immutable(bc.apply_contrast, test_image, value=0.5)

    def test_apply_contrast_zero(self, test_image):
        _check_immutable(bc.apply_contrast, test_image, value=0.0)

    def test_auto_contrast(self, test_image):
        _check_immutable(bc.auto_contrast, test_image, percentile_low=5.0, percentile_high=95.0)

    def test_to_grayscale(self, test_image):
        _check_immutable(bc.to_grayscale, test_image)


# ---------------------------------------------------------------------------
# Тести для hdr
# ---------------------------------------------------------------------------

class TestHdrImmutability:
    def test_apply(self, test_image):
        _check_immutable(hdr.apply, test_image, strength=0.5)

    def test_apply_zero(self, test_image):
        _check_immutable(hdr.apply, test_image, strength=0.0)

    def test_apply_max(self, test_image):
        _check_immutable(hdr.apply, test_image, strength=1.0)


# ---------------------------------------------------------------------------
# Тести для sharpen
# ---------------------------------------------------------------------------

class TestSharpenImmutability:
    def test_apply(self, test_image):
        _check_immutable(sharpen.apply, test_image, strength=0.4)

    def test_apply_zero(self, test_image):
        _check_immutable(sharpen.apply, test_image, strength=0.0)

    def test_measure_sharpness(self, test_image):
        """measure_sharpness повертає float, не масив — перевіряємо лише immutability."""
        before = test_image.copy()
        _ = sharpen.measure_sharpness(test_image)
        assert np.array_equal(test_image, before), (
            "sharpen.measure_sharpness змінив вхідне зображення!"
        )

    def test_auto_apply(self, test_image):
        _check_immutable(sharpen.auto_apply, test_image, threshold=80.0, max_strength=0.7)


# ---------------------------------------------------------------------------
# Тести для shadow_remove
# ---------------------------------------------------------------------------

class TestShadowRemoveImmutability:
    def test_remove_shadow(self, test_image):
        _check_immutable(shadow_remove.remove_shadow, test_image, kernel_size=0)

    def test_remove_shadow_fixed_kernel(self, test_image):
        _check_immutable(shadow_remove.remove_shadow, test_image, kernel_size=51)

    def test_auto_remove_shadow(self, test_image):
        _check_immutable(shadow_remove.auto_remove_shadow, test_image)


# ---------------------------------------------------------------------------
# Тести для shadow_highlight
# ---------------------------------------------------------------------------

class TestShadowHighlightImmutability:
    def test_apply_shadow_highlight(self, test_image):
        _check_immutable(shadow_highlight.apply_shadow_highlight, test_image, strength=0.5)

    def test_apply_shadow_highlight_zero(self, test_image):
        _check_immutable(shadow_highlight.apply_shadow_highlight, test_image, strength=0.0)

    def test_auto_shadow_highlight(self, test_image):
        _check_immutable(shadow_highlight.auto_shadow_highlight, test_image,
                        percentile_low=10.0, max_strength=0.6)


# ---------------------------------------------------------------------------
# Тести для perspective
# ---------------------------------------------------------------------------

class TestPerspectiveImmutability:
    def test_auto_detect_corners(self, test_image):
        """auto_detect_corners повертає кути або None — перевіряємо immutability."""
        before = test_image.copy()
        _ = perspective.auto_detect_corners(test_image)
        assert np.array_equal(test_image, before), (
            "perspective.auto_detect_corners змінив вхідне зображення!"
        )

    def test_apply_correction(self, test_image):
        """Тест з синтетичними кутами."""
        h, w = test_image.shape[:2]
        # Невелике зміщення — імітуємо перспективу
        corners = np.array([
            [50, 50],           # TL
            [w - 50, 30],       # TR
            [w - 30, h - 40],   # BR
            [40, h - 60],       # BL
        ], dtype=np.float32)
        _check_immutable(perspective.apply_correction, test_image, corners=corners)

    def test_auto_correct(self, test_image):
        _check_immutable(perspective.auto_correct, test_image)

    def test_order_points(self, test_image):
        """_order_points — внутрішня, але перевіримо, що не чіпає image."""
        before = test_image.copy()
        pts = np.array([[200, 100], [600, 80], [580, 500], [180, 520]], dtype=np.float32)
        _ = perspective._order_points(pts)
        assert np.array_equal(test_image, before), (
            "perspective._order_points змінив вхідне зображення!"
        )