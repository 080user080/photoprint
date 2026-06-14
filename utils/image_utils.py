"""
Утиліти для роботи із зображеннями (numpy/OpenCV).
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np
from collections import OrderedDict
from typing import Optional

# Константи для прев'ю
DEFAULT_PREVIEW_MAX_SIDE = 900

# LRU cache для прев'ю (ключ — id(path або image), значення — масив прев'ю)
# Обмеження: 20 елементів, щоб не тримати в пам'яті всі прев'ю великої черги
_PREVIEW_CACHE_MAX_SIZE = 20
_PREVIEW_CACHE: "OrderedDict[int, np.ndarray]" = OrderedDict()


def _preview_cache_key(image: np.ndarray, max_side: int) -> int:
    """
    Генерує унікальний ключ для кешу на основі shape, dtype та max_side.
    Не використовує id() безпосередньо, щоб кеш працював навіть якщо
    той самий масив було завантажено повторно.
    """
    return hash((image.shape, image.dtype, image.ctypes.data, max_side))


def preview_cache_clear() -> None:
    """Очищує весь кеш прев'ю. Викликати при закритті файлу або скиданні черги."""
    _PREVIEW_CACHE.clear()


def preview_cache_stats() -> dict:
    """Повертає статистику кешу: розмір, загальний об'єм у байтах."""
    total_bytes = sum(a.nbytes for a in _PREVIEW_CACHE.values())
    return {
        "size": len(_PREVIEW_CACHE),
        "max_size": _PREVIEW_CACHE_MAX_SIZE,
        "total_bytes": total_bytes,
        "total_mib": total_bytes / 1024 / 1024,
    }


def make_preview(image: np.ndarray, max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> np.ndarray:
    """
    Повертає зменшену копію для прев'ю.
    Оригінальне зображення не змінюється.
    Використовує LRU-кеш: повторні запити з тими ж (image.data, max_side)
    повертають кешований результат.
    """
    key = _preview_cache_key(image, max_side)

    # Якщо є в кеші — переміщуємо в кінець (як "нещодавно використаний")
    if key in _PREVIEW_CACHE:
        cached = _PREVIEW_CACHE.pop(key)
        _PREVIEW_CACHE[key] = cached
        return cached

    # Обчислюємо прев'ю
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        result = image.copy()
    else:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Додаємо в кеш
    _PREVIEW_CACHE[key] = result

    # LRU eviction — видаляємо найстаріший елемент, якщо перевищено ліміт
    while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_MAX_SIZE:
        _PREVIEW_CACHE.popitem(last=False)  # видаляємо перший (найстаріший)

    return result


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV → RGB для відображення у Qt."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Гарантує 3-канальний BGR (конвертує з grayscale якщо треба)."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
