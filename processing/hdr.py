"""
HDR tone mapping — локальне витягування деталей із тіней і світл.
Без злиття кількох знімків. Чисто numpy/OpenCV.
Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для HDR
HDR_THRESHOLD = 0.0  # мінімальна сила для застосування
HDR_MAX_STRENGTH = 1.0  # максимальна сила
HDR_CLIP_LIMIT_BASE = 1.0  # базовий clip limit
HDR_CLIP_LIMIT_MULTIPLIER = 3.0  # clip_limit = 1.0 + strength * 3.0
HDR_TILE_SIZE = 8  # розмір тайлу для CLAHE

# Константи для адаптивного HDR
ADAPTIVE_HDR_ALPHA_TEXT = 0.1  # коефіцієнт змішування HDR для текстових областей
ADAPTIVE_HDR_ALPHA_BACKGROUND = 1.0  # коефіцієнт змішування HDR для фону/однотонних областей
ADAPTIVE_HDR_BLEND_KERNEL = (15, 15)  # ядро для розмиття маски змішування (плавний перехід)


def apply(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Простий HDR ефект через CLAHE у каналі яскравості (LAB).
    strength: 0.0 – без ефекту, 1.0 – максимальне витягування деталей.
    Повертає uint8 BGR.
    """
    if strength <= HDR_THRESHOLD:
        return image.copy()

    strength = min(strength, HDR_MAX_STRENGTH)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE з адаптивним clip_limit залежно від strength
    clip_limit = HDR_CLIP_LIMIT_BASE + strength * HDR_CLIP_LIMIT_MULTIPLIER
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(HDR_TILE_SIZE, HDR_TILE_SIZE))
    l_eq = clahe.apply(l_ch)

    # Blend між оригінальним L і обробленим залежно від strength
    l_result = cv2.addWeighted(l_ch, 1.0 - strength, l_eq, strength, 0)

    merged = cv2.merge([l_result, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result


def _compute_hdr_lab(l_ch: np.ndarray, strength: float) -> np.ndarray:
    """
    Внутрішня: застосовує CLAHE до L-каналу та blend з оригіналом.
    Повертає оброблений L-канал.
    """
    clip_limit = HDR_CLIP_LIMIT_BASE + strength * HDR_CLIP_LIMIT_MULTIPLIER
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(HDR_TILE_SIZE, HDR_TILE_SIZE))
    l_eq = clahe.apply(l_ch)
    return cv2.addWeighted(l_ch, 1.0 - strength, l_eq, strength, 0)


def apply_adaptive(
    image: np.ndarray,
    strength: float = 0.5,
    text_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Адаптивний HDR: застосовує CLAHE до всього зображення, але зменшує
    ефект у текстових областях (щоб не затемнювати світлий фон навколо тексту).

    image: BGR uint8.
    strength: 0.0 – без ефекту, 1.0 – максимальне витягування деталей.
    text_mask: бінарна маска (uint8, 0/255) від text_mask.text_region_mask().
               Якщо None — поводиться як звичайний apply().

    Повертає uint8 BGR.
    """
    if strength <= HDR_THRESHOLD:
        return image.copy()

    if text_mask is None:
        return apply(image, strength=strength)

    strength = min(strength, HDR_MAX_STRENGTH)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Оброблений L-канал
    l_hdr = _compute_hdr_lab(l_ch, strength)

    # Створюємо маску змішування: розмиваємо text_mask для плавного переходу
    mask_float = text_mask.astype(np.float32) / 255.0  # 0..1, 1 = текст
    mask_blurred = cv2.GaussianBlur(mask_float, ADAPTIVE_HDR_BLEND_KERNEL, 0)
    # mask_blurred = 1.0 там, де точно текст; 0.0 — де фон

    # alpha = ADAPTIVE_HDR_ALPHA_TEXT для тексту, ADAPTIVE_HDR_ALPHA_BACKGROUND для фону
    alpha_map = ADAPTIVE_HDR_ALPHA_TEXT * mask_blurred + ADAPTIVE_HDR_ALPHA_BACKGROUND * (1.0 - mask_blurred)
    # l_result = l_ch + alpha * (l_hdr - l_ch)
    l_ch_f = l_ch.astype(np.float32)
    l_hdr_f = l_hdr.astype(np.float32)
    diff = l_hdr_f - l_ch_f
    l_result_f = l_ch_f + alpha_map * diff
    l_result = np.clip(l_result_f, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_result, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
