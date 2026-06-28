"""
Корекція кольорового відтінку фону документа.
Визначає домінантний відтінок світлого фону і нейтралізує його.
Працює незалежно від shadow_remove.
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Мінімальна яскравість пікселя щоб вважатись "фоном"
COLOR_CAST_BG_L_MIN = 170
# Мінімальна частка фонових пікселів для аналізу
COLOR_CAST_BG_MIN_RATIO = 0.10
# Мінімальне зміщення каналу для корекції (нижче — не чіпаємо)
COLOR_CAST_MIN_SHIFT = 3.0
# Максимальна сила корекції (захист від пересвічування)
COLOR_CAST_MAX_SHIFT = 25.0
# Сила блендингу (1.0 = повна корекція, 0.7 = 70%)
COLOR_CAST_BLEND = 0.85


def detect_color_cast(image: np.ndarray) -> tuple[float, float]:
    """
    Визначає відтінок фону документа (зміщення a та b каналів LAB).
    Аналізує тільки світлі ділянки (фон), ігнорує текст/печатки.
    
    Повертає (a_shift, b_shift) — наскільки треба зсунути канали до нейтрального.
    (0.0, 0.0) якщо відтінок не виявлено або він занадто малий.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    bg_mask = l_ch > COLOR_CAST_BG_L_MIN
    bg_ratio = float(np.mean(bg_mask))

    if bg_ratio < COLOR_CAST_BG_MIN_RATIO:
        return 0.0, 0.0

    a_median = float(np.median(a_ch[bg_mask]))
    b_median = float(np.median(b_ch[bg_mask]))

    # Нейтральний LAB: a=128, b=128
    a_shift = 128.0 - a_median
    b_shift = 128.0 - b_median

    # Ігноруємо дуже малі відхилення
    if abs(a_shift) < COLOR_CAST_MIN_SHIFT:
        a_shift = 0.0
    if abs(b_shift) < COLOR_CAST_MIN_SHIFT:
        b_shift = 0.0

    # Обмежуємо максимальну корекцію
    a_shift = float(np.clip(a_shift, -COLOR_CAST_MAX_SHIFT, COLOR_CAST_MAX_SHIFT))
    b_shift = float(np.clip(b_shift, -COLOR_CAST_MAX_SHIFT, COLOR_CAST_MAX_SHIFT))

    return a_shift, b_shift


def correct_color_cast(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Нейтралізує кольоровий відтінок фону документа.
    
    Алгоритм:
    1. Знаходить домінантний відтінок світлих ділянок (фону)
    2. Зсуває a та b канали LAB тільки на світлих ділянках
    3. На темних ділянках (текст, печатки) — не змінює
    
    Повертає (результат, чи_була_корекція).
    """
    a_shift, b_shift = detect_color_cast(image)

    if a_shift == 0.0 and b_shift == 0.0:
        return image.copy(), False

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    # Маска: тільки світлі ділянки отримують корекцію
    # Плавний перехід від 0 (темне) до 1 (світле) через 140-200 L
    weight = np.clip((l_ch.astype(np.float32) - 140.0) / 60.0, 0.0, 1.0)

    a_corrected = a_ch + a_shift * weight * COLOR_CAST_BLEND
    b_corrected = b_ch + b_shift * weight * COLOR_CAST_BLEND

    a_result = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_result = np.clip(b_corrected, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_ch, a_result, b_result])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result, True