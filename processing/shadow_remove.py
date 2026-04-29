"""
Видалення тіней з документів через background estimation.
Метод: GaussianBlur для оцінки фону + divide для нормалізації.
Прибирає градієнтні тіні (від папки, руки тощо) без перетворення їх на чорні плями.

Працює раніше за auto_contrast/CLAHE — тому наступні кроки бачать
"чистий" документ без тіней і не посилюють їх.

Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для background estimation
BLUR_KERNEL_MIN = 21       # мінімальний розмір ядра (непарне)
BLUR_KERNEL_MAX = 201      # максимальний розмір ядра
BLUR_KERNEL_STEP = 2       # крок збільшення (залишає непарним)
BLUR_SIGMA = 0             # 0 = автоматичний вибір sigma

# Константи для divide
DIVIDE_SCALE = 255.0       # масштаб для cv2.divide
DIVIDE_EPSILON = 1.0       # мінус до background щоб уникнути ділення на 0

# Константи для автоматичного визначення розміру ядра
KERNEL_SCALE_FACTOR = 5    # kernel = max(21, min_side // factor)
KERNEL_MIN_SIDE = 100      # мінімальний розмір сторони для масштабування

# Константи для детекції тіней
SHADOW_DETECT_PERCENTILE = 5   # нижній перцентиль для детекції
SHADOW_DETECT_THRESHOLD = 80   # поріг L-каналу: якщо p5 < threshold — є тіні
SHADOW_RATIO_THRESHOLD = 0.3  # мінімальне відношення p5/p95 для визнання тіней


def remove_shadow(image: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    """
    Видаляє градієнтні тіні з документа через background estimation.

    Метод:
    1. Сильно розмиваємо зображення → отримуємо "фон" (включно з тінню)
    2. Ділимо оригінал на фон → нормалізуємо освітлення
    3. Результат: рівномірно освітлений документ без тіней

    Args:
        image: BGR numpy array uint8
        kernel_size: Розмір ядра GaussianBlur (0 = автоматичний)
                     Чим більше ядро — тим більші тіні прибирає.
                     Має бути непарним.

    Returns:
        Оброблене BGR зображення без градієнтних тіней
    """
    if kernel_size == 0:
        kernel_size = _auto_kernel_size(image)

    # Гарантуємо непарність
    kernel_size = max(kernel_size | 1, BLUR_KERNEL_MIN)

    # Конвертуємо в LAB для роботи з каналом яскравості
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Оцінка фону через сильне розмиття
    background = cv2.GaussianBlur(l_ch, (kernel_size, kernel_size), BLUR_SIGMA)

    # Нормалізація: ділимо оригінал на фон
    # Конвертуємо в float32 для коректної роботи cv2.divide
    l_f = l_ch.astype(np.float32)
    bg_f = background.astype(np.float32) + DIVIDE_EPSILON
    l_norm = cv2.divide(l_f, bg_f, scale=DIVIDE_SCALE)
    l_norm = l_norm.astype(np.uint8)

    # Збираємо LAB назад
    merged = cv2.merge([l_norm, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result


def auto_remove_shadow(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Автоматичне видалення тіней: спочатку перевіряє чи є тіні,
    потім застосовує remove_shadow якщо потрібно.

    Повертає (результат, чи_були_тіні).
    """
    has_shadow = _detect_shadow(image)
    if not has_shadow:
        return image.copy(), False

    result = remove_shadow(image)
    return result, True


def _auto_kernel_size(image: np.ndarray) -> int:
    """
    Обчислює оптимальний розмір ядра на основі розміру зображення.
    Більше зображення — більше ядро (щоб розмиття "захопило" всю тінь).
    """
    h, w = image.shape[:2]
    min_side = min(h, w)

    if min_side < KERNEL_MIN_SIDE:
        return BLUR_KERNEL_MIN

    kernel = min_side // KERNEL_SCALE_FACTOR
    # Робимо непарним та обмежуємо
    kernel = kernel | 1  # гарантуємо непарність
    kernel = max(kernel, BLUR_KERNEL_MIN)
    kernel = min(kernel, BLUR_KERNEL_MAX)

    return kernel


def _detect_shadow(image: np.ndarray) -> bool:
    """
    Виявляє наявність тіней на зображенні.
    Якщо нижній перцентиль L-каналу значно темніший за верхній — є тіні.
    НЕ застосовуємо до фото (перевіряємо наявність кольору).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # Перевіряємо чи це фото (багато кольору) — не застосовуємо shadow_remove до фото
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    # Якщо багато кольору — це фото, не документ з тінню
    if std_a > 30 or std_b > 30:
        return False

    p_low = float(np.percentile(l, SHADOW_DETECT_PERCENTILE))
    p_high = float(np.percentile(l, 100 - SHADOW_DETECT_PERCENTILE))

    # Тіні є якщо:
    # 1. Нижній перцентиль дуже темний (p5 < 80)
    # 2. Різниця між низом та верхом значна (p5/p95 < 0.3)
    # 3. Діапазон не занадто великий (тінь vs рябий фон)
    if p_low < SHADOW_DETECT_THRESHOLD:
        ratio = p_low / max(p_high, 1.0)
        # Перевіряємо що діапазон не занадто широкий (рябий фон дає широкий діапазон)
        range_l = p_high - p_low
        if range_l > 200:  # Занадто широкий діапазон — скоріше рябий фон, не тінь
            return False
        return ratio < SHADOW_RATIO_THRESHOLD

    return False
