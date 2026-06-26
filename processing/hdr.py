"""
HDR tone mapping — локальное вытягивание деталей из теней и светов.
Без слияния нескольких снимков. Чисто numpy/OpenCV.
Не зависит от какого-либо другого модуля проекта.
"""

import cv2
import numpy as np

# Константы для HDR
HDR_THRESHOLD = 0.0  # минимальная сила для применения
HDR_MAX_STRENGTH = 1.0  # максимальная сила
HDR_CLIP_LIMIT_BASE = 1.0  # базовый clip limit
HDR_CLIP_LIMIT_MULTIPLIER = 1.5  # clip_limit = 1.0 + strength * 1.5 (меньше пятнистости)
HDR_TILE_SIZE = 8  # размер тайла для CLAHE (запасной, не используется в новых функциях)

# Константы для адаптивного HDR
ADAPTIVE_HDR_ALPHA_TEXT = 0.1  # коэффициент смешивания HDR для текстовых областей
ADAPTIVE_HDR_ALPHA_BACKGROUND = 1.0  # коэффициент смешивания HDR для фона/однотонных областей
ADAPTIVE_HDR_BLEND_KERNEL = (15, 15)  # ядро для размытия маски смешивания (плавный переход)

# --- Coring (подавление усиления шума) ---
HDR_NOISE_FLOOR = 1.5  # уровни L (0-255); diff меньше этого обнуляется
HDR_MANUAL_NOISE_FLOOR = 0.3   # знижений coring для ручного режиму

# --- Адаптивный tile grid (resolution-independent CLAHE) ---
HDR_TILE_TARGET_PX = 64   # желаемый размер тайла в пикселях
HDR_TILE_MIN_GRID = 4     # минимум тайлов по стороне
HDR_TILE_MAX_GRID = 16    # максимум тайлов по стороне

# --- Авто-масштабирование strength за динамическим диапазоном изображения ---
HDR_RANGE_REF = 120.0        # "типичный" диапазон p95-p5 для полной силы эффекта
HDR_RANGE_MIN_FACTOR = 0.25  # минимальный множитель для очень плоских изображений


def adaptive_tile_grid(h: int, w: int, target_px: int = HDR_TILE_TARGET_PX) -> tuple[int, int]:
    """
    Размер CLAHE-сетки в тайлах, соответствующий примерно target_px пикселям
    на тайл — одинаковое поведение на превью и на полноразмерном скане.
    """
    gx = int(np.clip(w // target_px, HDR_TILE_MIN_GRID, HDR_TILE_MAX_GRID))
    gy = int(np.clip(h // target_px, HDR_TILE_MIN_GRID, HDR_TILE_MAX_GRID))
    return (gx, gy)


def _apply_coring(diff: np.ndarray, noise_floor: float | None = None) -> np.ndarray:
    """
    Soft-threshold разницы HDR-CLAHE относительно оригинала.
    Разницы меньше noise_floor (типично — усиленный шум на ровном фоне)
    обнуляются; большие — уменьшаются на noise_floor (без резкого излома).

    Если noise_floor не передан — использует HDR_NOISE_FLOOR.
    """
    if noise_floor is None:
        noise_floor = HDR_NOISE_FLOOR
    mag = np.maximum(np.abs(diff) - noise_floor, 0.0)
    return np.sign(diff) * mag


def _auto_strength_factor(l_ch: np.ndarray) -> float:
    """
    Множитель 0..1 для strength, зависящий от собственного динамического
    диапазона изображения (p95-p5 канала L).

    Узкий диапазон (почти плоское/светлое изображение, типичный "белый фон")
    -> низкий фактор -> CLAHE получает меньший clip_limit -> меньше усиление
    шума. Широкий диапазон -> фактор ~1 -> полная сила, как задано слайдером.
    """
    p_low = float(np.percentile(l_ch, 5))
    p_high = float(np.percentile(l_ch, 95))
    range_l = p_high - p_low
    return float(np.clip(range_l / HDR_RANGE_REF, HDR_RANGE_MIN_FACTOR, 1.0))


def _compute_hdr_lab(l_ch: np.ndarray, strength: float) -> np.ndarray:
    """
    Внутренняя: применяет CLAHE к L-каналу и blend с оригиналом.
    Использует адаптивный tile grid и авто-масштабирование strength.
    Возвращает обработанный L-канал.
    """
    h, w = l_ch.shape[:2]
    factor = _auto_strength_factor(l_ch)
    effective_strength = strength * factor

    clip_limit = HDR_CLIP_LIMIT_BASE + effective_strength * HDR_CLIP_LIMIT_MULTIPLIER
    tile_grid = adaptive_tile_grid(h, w)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)

    return cv2.addWeighted(l_ch, 1.0 - effective_strength, l_eq, effective_strength, 0)


def apply(image: np.ndarray, strength: float = 0.5, manual_mode: bool = False) -> np.ndarray:
    """
    Простой HDR эффект через CLAHE в канале яркости (LAB).
    strength: 0.0 – без эффекта, 1.0 – максимальное вытягивание деталей.
    Возвращает uint8 BGR.

    Теперь является тонкой обёрткой над apply_adaptive(text_mask=None),
    что автоматически добавляет detail-mask + coring + auto-strength + adaptive tile grid.
    """
    if strength <= HDR_THRESHOLD:
        return image.copy()
    return apply_adaptive(image, strength=strength, text_mask=None, manual_mode=manual_mode)


def apply_adaptive(
    image: np.ndarray,
    strength: float = 0.5,
    text_mask: np.ndarray | None = None,
    auto_detail: bool = True,
    noise_floor: float | None = None,
    manual_mode: bool = False,
) -> np.ndarray:
    """
    Адаптивный HDR: применяет CLAKE ко всему изображению, но уменьшает
    эффект в текстовых областях (чтобы не затемнять светлый фон вокруг текста)
    и на однотонных участках (фикс артефактов на белом/сером фоне).

    image: BGR uint8.
    strength: 0.0 – без эффекта, 1.0 – максимальное вытягивание деталей.
    text_mask: бинарная маска (uint8, 0/255) от text_mask.text_region_mask().
               Если None — поводится как с детекцией однотонных участков.
    auto_detail: если True (дефолт) — дополнительно ограничивает эффект на однотонных
                 участках через processing.detail_map.detail_mask. Это и есть фикс
                 артефактов на белом/серо-светлом фоне. Работает независимо от text_mask.
    noise_floor: порог coring для подавления шума. Если None — используется HDR_NOISE_FLOOR.
    manual_mode: если True — пропускает _auto_strength_factor, пропускает detail_mask,
                 использует HDR_MANUAL_NOISE_FLOOR для coring.

    Возвращает uint8 BGR.
    """
    if strength <= HDR_THRESHOLD:
        return image.copy()

    strength = min(strength, HDR_MAX_STRENGTH)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Якщо manual_mode — не використовуємо _auto_strength_factor, тобто ефективна сила = strength
    if manual_mode:
        effective_strength = strength
    else:
        effective_strength = strength * _auto_strength_factor(l_ch)

    clip_limit = HDR_CLIP_LIMIT_BASE + effective_strength * HDR_CLIP_LIMIT_MULTIPLIER
    tile_grid = adaptive_tile_grid(l_ch.shape[0], l_ch.shape[1])
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)

    l_hdr = cv2.addWeighted(l_ch, 1.0 - effective_strength, l_eq, effective_strength, 0)

    l_ch_f = l_ch.astype(np.float32)
    l_hdr_f = l_hdr.astype(np.float32)

    # В ручному режимі використовуємо знижений noise_floor
    if manual_mode:
        nf = HDR_MANUAL_NOISE_FLOOR if noise_floor is None else noise_floor
    else:
        nf = noise_floor
    diff = _apply_coring(l_hdr_f - l_ch_f, noise_floor=nf)

    # --- alpha map: базово "полный эффект" везде ---
    alpha_map = np.full(l_ch.shape, ADAPTIVE_HDR_ALPHA_BACKGROUND, dtype=np.float32)

    # --- ограничение на однотонных участках (новый, основной фикс) ---
    # В ручному режимі пропускаємо detail_mask (ефект діє по всьому зображенню)
    if auto_detail and not manual_mode:
        from processing import detail_map as detail_map_module
        dmask = detail_map_module.detail_mask(l_ch, noise_floor=nf)  # 0..1, 0=плоско
        alpha_map *= dmask

    # --- существующее ограничение на тексте (как и раньше) ---
    if text_mask is not None:
        mask_float = text_mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, ADAPTIVE_HDR_BLEND_KERNEL, 0)
        alpha_map = alpha_map * (1.0 - mask_blurred) + ADAPTIVE_HDR_ALPHA_TEXT * mask_blurred

    l_result_f = l_ch_f + alpha_map * diff
    l_result = np.clip(l_result_f, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_result, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
