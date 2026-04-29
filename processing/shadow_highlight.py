"""
Висвітлення тіней (Shadow Highlight).
Працює в LAB color space для локального підвищення яскравості темних областей.
"""

import cv2
import numpy as np

# Константи для shadow highlight
SHADOW_THRESHOLD = 0.0  # мінімальна сила для застосування
SHADOW_L_THRESHOLD = 0.5  # поріг для визначення тіней (L < 0.5)
SHADOW_BOOST_MULTIPLIER = 2.0  # boost = 1.0 + strength * 2.0
L_NORMALIZATION = 255.0  # нормалізація L каналу

# Константи для авто shadow highlight
AUTO_SHADOW_PERCENTILE_LOW = 10.0  # нижній перцентиль за замовчуванням
AUTO_SHADOW_MAX_STRENGTH = 0.6  # максимальна сила за замовчуванням
AUTO_SHADOW_L_THRESHOLD = 50  # поріг для автоматичної корекції


def apply_shadow_highlight(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Висвітлює тіні на зображенні.

    Args:
        image: BGR numpy array uint8
        strength: Сила ефекту (0.0 - 2.0), де 0 = без змін, 2 = максимальне висвітлення

    Returns:
        Оброблене BGR зображення
    """
    if strength <= SHADOW_THRESHOLD:
        return image.copy()

    # Конвертуємо в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Нормалізуємо L канал до 0-1
    l_norm = l.astype(np.float32) / L_NORMALIZATION

    # Створюємо маску для темних областей (тіней)
    # Тіні - це області з L < SHADOW_L_THRESHOLD
    shadow_mask = 1.0 - (l_norm / SHADOW_L_THRESHOLD)
    shadow_mask = np.clip(shadow_mask, 0, 1)

    # Просте підвищення яскравості тіней - більш помітний ефект
    # Множник від 1.0 до 5.0 для діапазону 0.0-2.0
    boost = 1.0 + (strength * SHADOW_BOOST_MULTIPLIER)
    l_corrected = l_norm * boost

    # Змішуємо оригінал і корекцію на основі маски тіней
    l_final = l_norm * (1 - shadow_mask) + l_corrected * shadow_mask

    # Повертаємо до 0-255
    l_final = np.clip(l_final * L_NORMALIZATION, 0, L_NORMALIZATION).astype(np.uint8)

    # Збираємо LAB назад
    lab_corrected = cv2.merge([l_final, a, b])

    # Конвертуємо в BGR
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return result


def auto_shadow_highlight(image: np.ndarray, percentile_low: float = AUTO_SHADOW_PERCENTILE_LOW, max_strength: float = AUTO_SHADOW_MAX_STRENGTH) -> np.ndarray:
    """
    Автоматичне висвітлення тіней на основі гістограми.

    Args:
        image: BGR numpy array uint8
        percentile_low: Нижній перцентиль для визначення тіней
        max_strength: Максимальна сила ефекту

    Returns:
        Оброблене BGR зображення
    """
    # Конвертуємо в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    # Обчислюємо нижній перцентиль
    low_val = np.percentile(l, percentile_low)

    # Якщо тіні занадто темні, застосовуємо корекцію
    if low_val < AUTO_SHADOW_L_THRESHOLD:
        # Сила пропорційна темноті тіней
        strength = min((AUTO_SHADOW_L_THRESHOLD - low_val) / AUTO_SHADOW_L_THRESHOLD * max_strength, max_strength)
        return apply_shadow_highlight(image, strength)

    return image.copy()
