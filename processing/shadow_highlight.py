"""
Висвітлення тіней (Shadow Highlight).
Працює в LAB color space для локального підвищення яскравості темних областей.
"""

import cv2
import numpy as np


def apply_shadow_highlight(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Висвітлює тіні на зображенні.

    Args:
        image: BGR numpy array uint8
        strength: Сила ефекту (0.0 - 2.0), де 0 = без змін, 2 = максимальне висвітлення

    Returns:
        Оброблене BGR зображення
    """
    if strength <= 0:
        return image.copy()

    # Конвертуємо в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Нормалізуємо L канал до 0-1
    l_norm = l.astype(np.float32) / 255.0

    # Створюємо маску для темних областей (тіней)
    # Тіні - це області з L < 0.5
    shadow_mask = 1.0 - (l_norm / 0.5)
    shadow_mask = np.clip(shadow_mask, 0, 1)

    # Просте підвищення яскравості тіней - більш помітний ефект
    # Множник від 1.0 до 5.0 для діапазону 0.0-2.0
    boost = 1.0 + (strength * 2.0)
    l_corrected = l_norm * boost

    # Змішуємо оригінал і корекцію на основі маски тіней
    l_final = l_norm * (1 - shadow_mask) + l_corrected * shadow_mask

    # Повертаємо до 0-255
    l_final = np.clip(l_final * 255, 0, 255).astype(np.uint8)

    # Збираємо LAB назад
    lab_corrected = cv2.merge([l_final, a, b])

    # Конвертуємо в BGR
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return result


def auto_shadow_highlight(image: np.ndarray, percentile_low: float = 10.0, max_strength: float = 0.6) -> np.ndarray:
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
    if low_val < 50:
        # Сила пропорційна темноті тіней
        strength = min((50 - low_val) / 50.0 * max_strength, max_strength)
        return apply_shadow_highlight(image, strength)
    
    return image.copy()
