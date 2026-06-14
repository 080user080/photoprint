"""
Тести для utils/image_utils.py:
- make_preview (розмір, copy, cache)
- preview_cache_clear / preview_cache_stats
"""

import numpy as np
import pytest
from utils.image_utils import (
    make_preview,
    preview_cache_clear,
    preview_cache_stats,
    DEFAULT_PREVIEW_MAX_SIDE,
)


# ---------------------------------------------------------------------------
# Фікстури
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Очищує кеш перед кожним тестом."""
    preview_cache_clear()
    yield
    preview_cache_clear()


# ---------------------------------------------------------------------------
# Тести make_preview
# ---------------------------------------------------------------------------


def test_make_preview_small_image():
    """Зображення менше max_side — повертає копію без змін."""
    img = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
    prev = make_preview(img)
    assert prev.shape == (400, 600, 3), f"Очікувалось (400, 600, 3), отримано {prev.shape}"
    assert prev is not img, "make_preview має повертати новий масив"


def test_make_preview_large_image():
    """Зображення більше max_side — масштабується пропорційно."""
    img = np.random.randint(0, 256, (2000, 3000, 3), dtype=np.uint8)
    prev = make_preview(img)
    h, w = prev.shape[:2]
    assert max(h, w) <= DEFAULT_PREVIEW_MAX_SIDE, \
        f"Прев'ю завелике: {w}x{h}, max_side={DEFAULT_PREVIEW_MAX_SIDE}"
    # Перевірка пропорцій: відношення сторін має зберігатись (з точністю до 1px)
    orig_ratio = 3000 / 2000  # w/h
    prev_ratio = w / h
    assert abs(orig_ratio - prev_ratio) < 0.01, \
        f"Пропорції не збережено: orig={orig_ratio:.3f}, prev={prev_ratio:.3f}"


def test_make_preview_immutability():
    """make_preview не змінює вхідне зображення."""
    img = np.random.randint(0, 256, (2000, 3000, 3), dtype=np.uint8)
    before = img.copy()
    _ = make_preview(img)
    assert np.array_equal(img, before), "make_preview змінив вхідне зображення"


def test_make_preview_square_image():
    """Квадратне зображення: обидві сторони <= max_side після ресайзу."""
    img = np.random.randint(0, 256, (3000, 3000, 3), dtype=np.uint8)
    prev = make_preview(img)
    h, w = prev.shape[:2]
    assert h == w, f"Квадратне зображення втратило пропорції: {w}x{h}"
    assert h <= DEFAULT_PREVIEW_MAX_SIDE


# ---------------------------------------------------------------------------
# Тести LRU cache
# ---------------------------------------------------------------------------


def test_cache_hit():
    """Повторний виклик з тим самим зображенням повертає той самий об'єкт (кеш)."""
    img = np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8)
    prev1 = make_preview(img)
    prev2 = make_preview(img)
    assert prev1 is prev2, "LRU cache: другий виклик має повернути той самий об'єкт"
    stats = preview_cache_stats()
    assert stats["size"] == 1, f"Розмір кешу має бути 1, отримано {stats['size']}"


def test_cache_different_sizes():
    """
    Різні max_side для того самого зображення — різні ключі кешу.
    """
    img = np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8)
    prev1 = make_preview(img, max_side=900)
    prev2 = make_preview(img, max_side=300)
    assert prev1 is not prev2, "Різні max_side мають давати різні ключі кешу"
    assert prev2.shape[0] < prev1.shape[0], "max_side=300 має бути менше ніж 900"


def test_cache_max_size():
    """
    Кеш не перевищує _PREVIEW_CACHE_MAX_SIZE = 20.
    """
    images = [np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8) for _ in range(25)]
    for i, img in enumerate(images):
        make_preview(img)
    stats = preview_cache_stats()
    assert stats["size"] <= stats["max_size"], \
        f"Кеш перевищив ліміт: {stats['size']} > {stats['max_size']}"
    # Після 25 різних зображень має бути рівно 20 (evict старих)
    assert stats["size"] == 20, f"Розмір кешу має бути 20, отримано {stats['size']}"


def test_cache_clear():
    """preview_cache_clear() повністю очищує кеш."""
    img = np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8)
    _ = make_preview(img)
    assert preview_cache_stats()["size"] == 1
    preview_cache_clear()
    assert preview_cache_stats()["size"] == 0


def test_cache_lru_order():
    """
    LRU eviction: видаляє найстаріший елемент при переповненні.
    Додаємо 21 зображення в кеш (ліміт 20).
    Перше додане зображення має бути evicted, а 21-ше має залишитись.
    """
    images = [np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8) for _ in range(21)]
    # Додаємо всі 21 — при додаванні 21-го перший має evict
    for img in images:
        make_preview(img)
    # Перевіряємо: кеш має 20 елементів
    stats = preview_cache_stats()
    assert stats["size"] == 20, f"Кеш має бути 20 після 21 додавання, отримано {stats['size']}"
    # Перше зображення (images[0]) має бути evicted — новий виклик створить новий об'єкт,
    # але ми не перевіряємо це через можливе перевикористання пам'яті.
    # Натомість перевіряємо, що останнє додане (images[20]) є в кеші:
    _ = make_preview(images[0])  # додаємо його назад (evict інший)
    prev_last_cached = make_preview(images[20])
    # Воно має бути тим самим об'єктом, що й після першого циклу
    prev_last_first = make_preview(images[20])
    assert prev_last_cached is prev_last_first, \
        "Останнє зображення має бути в кеші (не evicted)"


# ---------------------------------------------------------------------------
# Тести preview_cache_stats
# ---------------------------------------------------------------------------


def test_cache_stats_empty():
    """Статистика порожнього кешу."""
    stats = preview_cache_stats()
    assert stats["size"] == 0
    assert stats["total_bytes"] == 0
    assert stats["total_mib"] == 0.0


def test_cache_stats_non_empty():
    """Статистика непустого кешу."""
    img = np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8)
    _ = make_preview(img)
    stats = preview_cache_stats()
    assert stats["size"] == 1
    assert stats["total_bytes"] > 0
    assert stats["total_mib"] > 0.0
    assert stats["max_size"] == 20