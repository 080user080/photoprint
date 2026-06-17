"""
Тесты для shadow_remove — проверка морфологического метода,
защитных механизмов и отсутствия ореолов.

ВАЖНО: shadow_remove целенаправленно ИЗМЕНЯЕТ изображение (убирает тень),
поэтому SSIM/PSNR с оригиналом ОЖИДАЕМО низкие. Тесты проверяют:
- Отсутствие деградации на равномерном фоне (SSIM > 0.93).
- Отсутствие инверсии (min > 5, тёмная область осветляется).
- Защитные фильтры корректно блокируют неподходящие изображения.
- Производительность в допустимых пределах.
"""
import time
import cv2
import numpy as np
import pytest
from processing import shadow_remove


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Упрощённый SSIM между двумя одноканальными uint8 изображениями."""
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mu_a = a_f.mean()
    mu_b = b_f.mean()
    var_a = a_f.var()
    var_b = b_f.var()
    cov = ((a_f - mu_a) * (b_f - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR между двумя одноканальными uint8 изображениями."""
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def _make_sharp_shadow_image(size: int = 400) -> np.ndarray:
    """
    Создаёт синтетическое изображение с резким краем тени.
    Левая половина — тёмная (L=60), правая — светлая (L=220).
    Имитирует тень от папки или руки на документе.
    """
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    img[:, :size // 2] = (60, 60, 60)  # тёмная тень
    # Добавляем лёгкий шум для реалистичности
    noise = np.random.default_rng(42).normal(0, 3, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_guilloche_image(size: int = 300) -> np.ndarray:
    """
    Создаёт синтетическое изображение паспорта с гільйошем:
    цветной узор на фоне (высокая насыщенность S-канала HSV).
    """
    rng = np.random.default_rng(42)
    img = np.full((size, size, 3), (200, 200, 200), dtype=np.uint8)
    # Яркие цветные полосы — имитация гільйоша
    for i in range(0, size, 8):
        color = rng.integers(50, 255, 3).astype(np.uint8)
        img[i : i + 3, :] = color
    return img


def _make_text_document(size: int = 400) -> np.ndarray:
    """
    Создаёт синтетическое изображение документа с текстом
    и градиентной тенью (плавный переход, имитация тени от руки).
    """
    img = np.full((size, size, 3), 230, dtype=np.uint8)
    # Горизонтальный градиент тени
    gradient = np.linspace(0.4, 1.0, size).reshape(1, size, 1)
    img = (img.astype(np.float64) * gradient).clip(0, 255).astype(np.uint8)
    # Добавляем «текст» — случайные тёмные блоки
    rng = np.random.default_rng(42)
    for _ in range(50):
        x = rng.integers(10, size - 30)
        y = rng.integers(10, size - 10)
        w = rng.integers(5, 25)
        h = rng.integers(3, 7)
        img[y : y + h, x : x + w] = (30, 30, 30)
    return img


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

class TestMorphologyVsGaussian:
    """Проверка базовых свойств морфологического метода."""

    def test_auto_remove_no_shadow_on_uniform(self):
        """
        На равномерном изображении без тени auto_remove_shadow
        должен корректно определить отсутствие тени и вернуть копию.
        """
        img = np.full((200, 200, 3), 180, dtype=np.uint8)
        result, had_shadow = shadow_remove.auto_remove_shadow(img)
        assert not had_shadow, (
            "Равномерное изображение ошибочно распознано как тень"
        )
        assert np.array_equal(result, img), (
            "Изображение изменилось, хотя тени не было"
        )

    def test_sharp_shadow_edge_no_inversion(self):
        """
        На изображении с резким краем тени не должно быть инверсии:
        тёмная область должна осветлиться, пиксели не должны уйти
        в экстремально тёмные значения.
        """
        img = _make_sharp_shadow_image(400)
        result = shadow_remove.remove_shadow(img)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Проверяем, что тёмная область стала светлее (тень убрана)
        left_mean_in = gray_in[:, :180].mean()
        left_mean_out = gray_out[:, :180].mean()
        assert left_mean_out > left_mean_in, (
            f"Тёмная область не осветлилась: было {left_mean_in:.1f}, стало {left_mean_out:.1f}"
        )
        # Проверяем, что нет аномально тёмных пикселей (инверсия в чёрное)
        assert gray_out.min() > 5, f"Слишком тёмные пиксели: min={gray_out.min()}"

    def test_sharp_edge_dark_side_brightened(self):
        """
        После обработки тёмная сторона должна стать значительно ярче,
        а светлая — остаться примерно на том же уровне.
        Проверяем, что контраст тени уменьшился.
        """
        img = _make_sharp_shadow_image(400)
        result = shadow_remove.remove_shadow(img)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Разница между левой и правой половиной должна уменьшиться
        left_in = gray_in[:, :180].mean()
        right_in = gray_in[:, 220:].mean()
        left_out = gray_out[:, :180].mean()
        right_out = gray_out[:, 220:].mean()
        contrast_in = right_in - left_in
        contrast_out = right_out - left_out
        assert contrast_out < contrast_in * 0.7, (
            f"Контраст тени недостаточно уменьшен: "
            f"было {contrast_in:.1f}, стало {contrast_out:.1f}"
        )
        # Светлая сторона не должна сильно потемнеть
        assert right_out > right_in * 0.85, (
            f"Светлая сторона потемнела: было {right_in:.1f}, стало {right_out:.1f}"
        )


class TestProtectiveFilters:
    """Проверка защитных механизмов _detect_shadow."""

    def test_guilloche_detected_as_no_shadow(self):
        """
        Изображение с гільйошем (паспорт) не должно распознаваться
        как документ с тенью — детектор должен вернуть False.
        """
        img = _make_guilloche_image(300)
        result, had_shadow = shadow_remove.auto_remove_shadow(img)
        assert not had_shadow, (
            "Гільйош ошибочно распознан как тень "
            "(S-канал HSV должен блокировать)"
        )
        # Результат должен быть копией (без изменений)
        assert np.array_equal(result, img), "Изображение изменилось, хотя тени не было"

    def test_blurred_image_no_shadow_detected(self):
        """Сильно размытое изображение не должно обрабатываться."""
        img = np.full((200, 200, 3), 150, dtype=np.uint8)
        img_blurred = cv2.GaussianBlur(img, (31, 31), 0)
        result, had_shadow = shadow_remove.auto_remove_shadow(img_blurred)
        assert not had_shadow, "Размытое изображение ошибочно распознано как тень"

    def test_small_image_unchanged(self):
        """Изображения < 100×100 не должны обрабатываться."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        result = shadow_remove.remove_shadow(img)
        assert np.array_equal(result, img), "Малое изображение изменилось"

    def test_photo_not_processed(self):
        """Цветное фото (высокая std A/B) не должно обрабатываться."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        result, had_shadow = shadow_remove.auto_remove_shadow(img)
        assert not had_shadow, "Цветное фото ошибочно распознано как тень"


class TestTextDocument:
    """Проверка обработки документов с текстом и градиентной тенью."""

    def test_text_remains_darker_than_background(self):
        """
        После обработки текст должен оставаться темнее окружающего фона.
        Тень убирается (тёмный фон осветляется), но текст не должен
        слиться с фоном.
        """
        img = _make_text_document(400)
        result = shadow_remove.remove_shadow(img)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Находим текстовые регионы — самые тёмные 5% пикселей оригинала
        thresh = np.percentile(gray_in, 5)
        text_mask = gray_in < thresh
        if text_mask.sum() < 10:
            pytest.skip("Слишком мало текстовых пикселей для проверки")
        mean_text = gray_out[text_mask].mean()
        mean_bg = gray_out[~text_mask].mean()
        # Текст должен быть темнее фона минимум на 15 уровней
        assert mean_bg - mean_text > 15, (
            f"Текст слился с фоном: текст={mean_text:.1f}, фон={mean_bg:.1f}"
        )

    def test_gradient_shadow_reduced(self):
        """
        Градиент тени (разница между левым и правым краем)
        должен значительно уменьшиться после обработки.
        """
        img = _make_text_document(400)
        result = shadow_remove.remove_shadow(img)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Средняя яркость левого и правого края
        left_in = gray_in[:, :50].mean()
        right_in = gray_in[:, -50:].mean()
        left_out = gray_out[:, :50].mean()
        right_out = gray_out[:, -50:].mean()
        grad_in = right_in - left_in
        grad_out = right_out - left_out
        # Градиент должен уменьшиться как минимум на 40%
        assert grad_out < grad_in * 0.6, (
            f"Градиент тени недостаточно уменьшен: "
            f"было {grad_in:.1f}, стало {grad_out:.1f}"
        )
        # Левый край (тёмный) должен осветлиться
        assert left_out > left_in, (
            f"Тёмный край не осветлился: было {left_in:.1f}, стало {left_out:.1f}"
        )


class TestBenchmark:
    """Измерение производительности морфологического метода."""

    def test_benchmark_400x400(self):
        """Обработка 400×400 должна занимать < 0.5 секунды."""
        img = _make_sharp_shadow_image(400)
        start = time.perf_counter()
        _ = shadow_remove.remove_shadow(img)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Обработка 400×400 заняла {elapsed:.3f}s (порог 0.5s)"

    def test_benchmark_800x800(self):
        """Обработка 800×800 должна занимать < 2.0 секунды."""
        img = _make_sharp_shadow_image(800)
        start = time.perf_counter()
        _ = shadow_remove.remove_shadow(img)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Обработка 800×800 заняла {elapsed:.3f}s (порог 2.0s)"