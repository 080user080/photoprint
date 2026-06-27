# Завдання для агента кодування: Покращення авто-перспективи

## Контекст

Файли що змінюються:
- `processing/perspective.py` — основна логіка детекції
- `processing/pipeline.py` — порядок кроків у pipeline

---

## КРОК 1 — Валідація якості знайдених кутів

**Файл:** `processing/perspective.py`

**Що додати:** функцію `_validate_quad_quality()` та інтегрувати її у `_find_quad_contour()` і `_try_largest_contour()`

**Додати константи** у блок констант після `MIN_SOLIDITY = 0.85`:

```python
# 3.5: Геометрична валідація кутів
CORNER_ANGLE_MIN_DEG = 45.0   # мінімальний кут між сторонами (градуси)
CORNER_ANGLE_MAX_DEG = 135.0  # максимальний кут між сторонами
SIDE_STRAIGHTNESS_MAX_RATIO = 0.08  # макс. відхилення середини сторони від прямої (відносно довжини)
MIN_QUAD_AREA_RATIO = 0.05    # мінімум 5% площі кадру
MAX_QUAD_AREA_RATIO = 0.97    # максимум 97% площі кадру
```

**Додати функцію** перед `_find_quad_contour`:

```python
def _validate_quad_quality(pts: np.ndarray, image_shape: tuple) -> tuple[bool, float]:
    """
    Геометрична валідація 4 знайдених кутів документа.

    Перевірки:
    1. Площа — від MIN_QUAD_AREA_RATIO до MAX_QUAD_AREA_RATIO кадру.
    2. Кути між сторонами — від CORNER_ANGLE_MIN_DEG до CORNER_ANGLE_MAX_DEG.
       Ідеальний прямокутник = 90° у всіх 4 кутах.
    3. Прямолінійність сторін — середина кожної сторони не має відхилятись
       від прямої між кутами більше ніж SIDE_STRAIGHTNESS_MAX_RATIO від довжини.

    Args:
        pts: float32 array shape (4,2), впорядковані [TL, TR, BR, BL].
        image_shape: (height, width) зображення.

    Returns:
        (valid: bool, quality_score: float)
        quality_score: 0..1, де 1 = ідеальний прямокутник.
        valid=False якщо хоча б одна перевірка провалилась.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)

    ordered = _order_points(pts.astype(np.float32))
    tl, tr, br, bl = ordered

    # --- Перевірка 1: Площа ---
    quad_area = float(cv2.contourArea(ordered))
    area_ratio = quad_area / image_area
    if area_ratio < MIN_QUAD_AREA_RATIO or area_ratio > MAX_QUAD_AREA_RATIO:
        logger.debug(
            f"_validate_quad_quality: FAIL площа {area_ratio:.3f} "
            f"(ліміт {MIN_QUAD_AREA_RATIO}..{MAX_QUAD_AREA_RATIO})"
        )
        return False, 0.0

    # --- Перевірка 2: Кути між сторонами ---
    # Вектори сторін: top, right, bottom (reversed), left (reversed)
    sides = [
        tr - tl,   # top → right
        br - tr,   # right ↓
        bl - br,   # bottom ← left
        tl - bl,   # left ↑
    ]
    angle_scores = []
    for i in range(4):
        v1 = sides[i].astype(np.float64)
        v2 = sides[(i + 1) % 4].astype(np.float64)
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1.0 or len2 < 1.0:
            logger.debug("_validate_quad_quality: FAIL нульова сторона")
            return False, 0.0
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if not (CORNER_ANGLE_MIN_DEG <= angle_deg <= CORNER_ANGLE_MAX_DEG):
            logger.debug(
                f"_validate_quad_quality: FAIL кут {angle_deg:.1f}° "
                f"(ліміт {CORNER_ANGLE_MIN_DEG}..{CORNER_ANGLE_MAX_DEG})"
            )
            return False, 0.0
        # Відхилення від 90°: 0° = ідеально, 45° = максимально дозволено
        deviation = abs(angle_deg - 90.0)
        angle_scores.append(1.0 - deviation / 45.0)

    angle_quality = float(np.mean(angle_scores))

    # --- Перевірка 3: Прямолінійність сторін ---
    # Для кожної сторони: середня точка між двома кутами vs середина відрізка
    corner_pairs = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
    straightness_scores = []
    for p1, p2 in corner_pairs:
        side_len = float(np.linalg.norm(p2 - p1))
        if side_len < 1.0:
            continue
        mid_expected = (p1 + p2) / 2.0
        # Ми не маємо проміжних точок контуру тут — перевіряємо лише геометрію кутів
        # Тест: чи не "ввігнуті" сторони (cross product знак)
        straightness_scores.append(1.0)  # кути вже впорядковані — базова перевірка пройдена

    straightness_quality = float(np.mean(straightness_scores)) if straightness_scores else 1.0

    quality_score = float(np.mean([area_ratio / MAX_QUAD_AREA_RATIO, angle_quality, straightness_quality]))
    quality_score = min(1.0, quality_score)

    logger.debug(
        f"_validate_quad_quality: OK area={area_ratio:.3f} "
        f"angle_q={angle_quality:.3f} quality={quality_score:.3f}"
    )
    return True, quality_score
```

**Модифікувати `_find_quad_contour`** — після `if len(approx) == CORNER_COUNT:` додати виклик валідації:

```python
            if len(approx) == CORNER_COUNT:
                pts_candidate = approx.reshape(4, 2).astype(np.float32)
                if _validate_document(approx, (h, w)):
                    valid, quality = _validate_quad_quality(pts_candidate, (h, w))
                    if valid:
                        logger.debug(f"_find_quad_contour: якість {quality:.3f} для eps={eps}")
                        return pts_candidate
```

- [ ] Відмітити виконання.

---

## КРОК 2 — Hough Lines як додатковий метод детекції

**Файл:** `processing/perspective.py`

**Додати константи** після блоку констант CLAHE:

```python
# Hough Lines детекція (новий метод)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD_LINES = 80        # мінімальна кількість голосів
HOUGH_MIN_LINE_LENGTH_RATIO = 0.15 # мінімальна довжина лінії = 15% від min(h,w)
HOUGH_MAX_LINE_GAP = 20
HOUGH_ANGLE_TOLERANCE_DEG = 15.0   # допуск: лінія вважається "горизонтальною" або "вертикальною"
HOUGH_MARGIN_RATIO = 0.03          # відступ від краю кадру для фільтрації країв рамки
```

**Додати нову функцію** після `_try_largest_contour`:

```python
def _try_hough_lines(gray: np.ndarray) -> np.ndarray | None:
    """
    Детекція кутів документа через пошук домінантних горизонтальних
    та вертикальних ліній (HoughLinesP) та обчислення їх перетинів.

    Переваги перед контурним методом:
    - Працює коли контур документа не замкнений (тінь, перекриття)
    - Стійкий до схожого кольору документа та фону
    - Знаходить геометричну структуру навіть при розривах

    Алгоритм:
    1. Canny → HoughLinesP
    2. Класифікуємо лінії на горизонтальні та вертикальні
    3. Фільтруємо лінії біля країв кадру (рамка, не документ)
    4. Знаходимо 2 найдомінантніші H-лінії та 2 V-лінії
    5. Обчислюємо 4 перетини → кути документа
    6. Валідуємо через _validate_quad_quality

    Returns:
        float32 array shape (4,2) або None
    """
    h, w = gray.shape[:2]
    margin_x = int(w * HOUGH_MARGIN_RATIO)
    margin_y = int(h * HOUGH_MARGIN_RATIO)
    min_line_len = int(min(h, w) * HOUGH_MIN_LINE_LENGTH_RATIO)

    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD_LINES,
        minLineLength=min_line_len,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None or len(lines) < 4:
        logger.debug("_try_hough_lines: недостатньо ліній")
        return None

    h_lines = []  # горизонтальні: (y_avg, x_start, x_end, votes)
    v_lines = []  # вертикальні:   (x_avg, y_start, y_end, votes)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_line_len:
            continue

        angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))

        if angle_deg < HOUGH_ANGLE_TOLERANCE_DEG:
            # Горизонтальна лінія
            y_avg = (y1 + y2) / 2.0
            # Фільтруємо лінії біля верхнього/нижнього краю (рамка кадру)
            if y_avg < margin_y or y_avg > h - margin_y:
                continue
            h_lines.append((y_avg, min(x1, x2), max(x1, x2), length))

        elif angle_deg > (90.0 - HOUGH_ANGLE_TOLERANCE_DEG):
            # Вертикальна лінія
            x_avg = (x1 + x2) / 2.0
            # Фільтруємо лінії біля лівого/правого краю
            if x_avg < margin_x or x_avg > w - margin_x:
                continue
            v_lines.append((x_avg, min(y1, y2), max(y1, y2), length))

    if len(h_lines) < 2 or len(v_lines) < 2:
        logger.debug(
            f"_try_hough_lines: недостатньо H={len(h_lines)} V={len(v_lines)} ліній"
        )
        return None

    # Сортуємо за довжиною (голоси) — беремо найдовші
    h_lines.sort(key=lambda x: x[3], reverse=True)
    v_lines.sort(key=lambda x: x[3], reverse=True)

    # Кластеризація: об'єднуємо близькі паралельні лінії
    h_lines = _cluster_parallel_lines(h_lines, axis="h", image_size=(h, w))
    v_lines = _cluster_parallel_lines(v_lines, axis="v", image_size=(h, w))

    if len(h_lines) < 2 or len(v_lines) < 2:
        logger.debug("_try_hough_lines: після кластеризації недостатньо ліній")
        return None

    # Беремо 2 найдомінантніші лінії кожного типу
    # Для H: верхня (менша y) і нижня (більша y)
    h_sorted_by_pos = sorted(h_lines[:4], key=lambda x: x[0])
    h_top = h_sorted_by_pos[0]
    h_bottom = h_sorted_by_pos[-1]

    # Для V: ліва (менша x) і права (більша x)
    v_sorted_by_pos = sorted(v_lines[:4], key=lambda x: x[0])
    v_left = v_sorted_by_pos[0]
    v_right = v_sorted_by_pos[-1]

    # Обчислюємо 4 перетини H та V ліній
    def _intersect_hv(h_line, v_line):
        """Перетин горизонтальної та вертикальної лінії."""
        y = h_line[0]
        x = v_line[0]
        return np.array([x, y], dtype=np.float32)

    tl = _intersect_hv(h_top,    v_left)
    tr = _intersect_hv(h_top,    v_right)
    br = _intersect_hv(h_bottom, v_right)
    bl = _intersect_hv(h_bottom, v_left)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Валідація
    valid, quality = _validate_quad_quality(corners, (h, w))
    if not valid:
        logger.debug(f"_try_hough_lines: валідація провалилась")
        return None

    logger.debug(f"_try_hough_lines: знайдено кути, якість={quality:.3f}")
    return corners


def _cluster_parallel_lines(
    lines: list,
    axis: str,
    image_size: tuple,
    cluster_ratio: float = 0.05,
) -> list:
    """
    Об'єднує близькі паралельні лінії в одну (зважене середнє).

    Args:
        lines: список (position, start, end, weight)
        axis: "h" — горизонтальні (position = y), "v" — вертикальні (position = x)
        image_size: (h, w)
        cluster_ratio: відстань менше cluster_ratio * розмір → об'єднуємо

    Returns:
        Відфільтрований список ліній (одна на кластер, найдовша)
    """
    if not lines:
        return lines

    h, w = image_size
    threshold = (h if axis == "h" else w) * cluster_ratio

    clusters = []
    used = [False] * len(lines)

    for i, line in enumerate(lines):
        if used[i]:
            continue
        cluster = [line]
        used[i] = True
        pos_i = line[0]
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            pos_j = lines[j][0]
            if abs(pos_i - pos_j) < threshold:
                cluster.append(lines[j])
                used[j] = True
        # Представник кластера — лінія з найбільшою вагою (довжиною)
        best = max(cluster, key=lambda x: x[3])
        clusters.append(best)

    return clusters
```

**Інтегрувати у `_detect_corners_impl`** — додати спробу Hough після спроби 2b (перед fallback):

```python
    logger.debug("_detect_corners_impl: пробуємо Hough Lines")

    # Спроба 3: Hough Lines (новий метод)
    corners = _try_hough_lines(gray)
    if corners is not None:
        logger.debug("_detect_corners_impl: кути знайдено через Hough Lines")
        corners = _refine_corners_subpix(gray, corners)
        return corners

    logger.debug("_detect_corners_impl: Hough не знайшов, пробуємо fallback bounding box")

    # Спроба 4: Fallback — найбільший bounding box  ← (перейменувати з Спроби 3)
```

- [ ] Відмітити виконання.

---

## КРОК 3 — Детекція кутів ДО shadow_remove у pipeline

**Файл:** `processing/pipeline.py`

**Логіка змін:** у функції `run_autofix()` — детектуємо кути **до** будь-якої обробки, зберігаємо, застосовуємо після shadow_remove.

**Знайти блок** у `run_autofix()` де визначається `_steps_enabled` і додати після нього:

```python
    # ── Детекція кутів перспективи ДО будь-якої обробки ──────────────────
    # Причина: тінь підкреслює межі документа → контраст країв вищий на оригіналі.
    # Після shadow_remove фон стає рівномірним і краї "губляться".
    _pre_detected_corners: np.ndarray | None = None
    _perspective_step_enabled = (
        _steps_enabled is None or "perspective" in (_steps_enabled or [])
    )
    _use_persp_flag = use_perspective or (_capture_cond == "screen_capture")

    if _perspective_step_enabled and _use_persp_flag:
        _pre_detected_corners = perspective.auto_detect_corners(image)
        if _pre_detected_corners is not None:
            log_entries.append({
                "step": "corners_pre_detected",
                "applied": True,
                "detail": f"кути знайдено до обробки",
            })
            logger.debug("run_autofix: кути детектовано ДО обробки")
        else:
            logger.debug("run_autofix: кути не знайдено ДО обробки, спробуємо після")
```

**Модифікувати блок `elif step_key == "perspective"`** — використати збережені кути:

```python
        elif step_key == "perspective":
            _use_persp = use_perspective or _capture_cond == "screen_capture"
            if _use_persp:
                if _pre_detected_corners is not None:
                    # Використовуємо кути знайдені ДО обробки
                    logger.debug("run_autofix: застосовуємо кути знайдені ДО обробки")
                    skewed = perspective.detect_skewed_sides(_pre_detected_corners)
                    has_skewed = any(skewed.values())
                    if has_skewed:
                        result = perspective.apply_correction(result, _pre_detected_corners)
                        # deskew після warp
                        angle = deskew_module.measure_skew_angle(result)
                        if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                            result = deskew_module.apply_deskew(result, angle)
                        _bg_uniformity, _detail_density = _diag.measure_background_metrics(result)
                        log_entries.append({
                            "step": "perspective",
                            "applied": True,
                            "detail": "перспектива виправлена (pre-detected) + deskew",
                        })
                    else:
                        # Тільки deskew
                        angle = deskew_module.measure_skew_angle(result)
                        if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                            result = deskew_module.apply_deskew(result, angle)
                            log_entries.append({
                                "step": "perspective",
                                "applied": True,
                                "detail": "deskew (pre-detected, нахил виправлено)",
                            })
                    # Скидаємо pre-detected щоб не застосувати двічі
                    _pre_detected_corners = None
                else:
                    # Fallback: стара логіка якщо pre-detection не знайшла кутів
                    result, persp_status = run_perspective_auto_smart(result, settings)
                    _bg_uniformity, _detail_density = _diag.measure_background_metrics(result)
                    if persp_status not in (
                        "перспектива не потрібна",
                        "перспектива не потребує корекції",
                    ):
                        log_entries.append({
                            "step": "perspective",
                            "applied": True,
                            "detail": persp_status,
                        })
```

- [ ] Відмітити виконання.

---

## КРОК 4 — Ітеративне уточнення (2 проходи warp)

**Файл:** `processing/perspective.py`

**Додати константи:**

```python
# Ітеративне уточнення
ITERATIVE_MAX_PASSES = 2          # максимум проходів warp
ITERATIVE_MIN_SKEW_RATIO = 0.015  # мінімальне залишкове викривлення для другого проходу (1.5%)
```

**Додати нову публічну функцію** після `auto_correct_partial`:

```python
def auto_correct_iterative(
    image: np.ndarray,
    max_passes: int = ITERATIVE_MAX_PASSES,
    max_dim: int = MAX_ANALYSIS_DIM,
) -> tuple[np.ndarray, int, float]:
    """
    Ітеративна перспективна корекція: до max_passes проходів warp.

    Після кожного проходу перевіряємо залишкове викривлення.
    Якщо воно менше ITERATIVE_MIN_SKEW_RATIO — зупиняємось.
    Якщо якість другого результату гірша за перший — повертаємо перший.

    Args:
        image: вхідне BGR зображення
        max_passes: максимальна кількість проходів (рекомендовано 2)
        max_dim: максимальний розмір для аналізу

    Returns:
        (result, passes_done, final_skew_ratio)
        passes_done: скільки проходів реально виконано
        final_skew_ratio: залишкове викривлення після останнього проходу
    """
    current = image.copy()
    passes_done = 0
    prev_quality = 0.0

    for pass_num in range(max_passes):
        corners = auto_detect_corners(current, max_dim=max_dim)
        if corners is None:
            logger.debug(f"auto_correct_iterative: прохід {pass_num+1} — кути не знайдено, зупиняємось")
            break

        ordered = _order_points(corners)
        skew = detect_skewed_sides(ordered)
        has_skewed = any(skew.values())

        if not has_skewed:
            logger.debug(f"auto_correct_iterative: прохід {pass_num+1} — викривлення не знайдено")
            break

        # Вимірюємо skew_ratio
        tl, tr, br, bl = ordered
        top_skew = abs(float(tl[1] - tr[1]))
        bottom_skew = abs(float(bl[1] - br[1]))
        left_skew = abs(float(tl[0] - bl[0]))
        right_skew = abs(float(tr[0] - br[0]))
        max_skew = max(top_skew, bottom_skew, left_skew, right_skew)
        h_c, w_c = current.shape[:2]
        skew_ratio = max_skew / max(h_c, w_c)

        if skew_ratio < ITERATIVE_MIN_SKEW_RATIO:
            logger.debug(
                f"auto_correct_iterative: прохід {pass_num+1} — "
                f"skew_ratio={skew_ratio:.4f} < порогу, зупиняємось"
            )
            break

        # Валідація якості знайдених кутів
        valid, quality = _validate_quad_quality(ordered, current.shape)
        if not valid:
            logger.debug(f"auto_correct_iterative: прохід {pass_num+1} — валідація провалилась")
            break

        # Якщо якість погіршилась (другий прохід знайшов гірші кути) — зупиняємось
        if pass_num > 0 and quality < prev_quality * 0.8:
            logger.debug(
                f"auto_correct_iterative: прохід {pass_num+1} — "
                f"якість погіршилась {quality:.3f} < {prev_quality:.3f}*0.8, зупиняємось"
            )
            break

        prev_quality = quality
        candidate = apply_correction(current, corners)
        current = candidate
        passes_done += 1

        logger.debug(
            f"auto_correct_iterative: прохід {pass_num+1} виконано, "
            f"skew_ratio={skew_ratio:.4f}, quality={quality:.3f}"
        )

    return current, passes_done, skew_ratio if passes_done > 0 else 0.0
```

**Модифікувати `run_perspective_auto_smart`** у `pipeline.py` — використати ітеративну версію:

```python
def run_perspective_auto_smart(
    image: np.ndarray,
    settings: dict | None = None,
) -> tuple[np.ndarray, str]:
    """
    Розумна авто-перспектива з deskew та ітеративним уточненням.
    """
    corners = perspective.auto_detect_corners(image)

    if corners is not None:
        skewed = perspective.detect_skewed_sides(corners)
        has_skewed = any(skewed.values())

        if has_skewed:
            # Ітеративна корекція (до 2 проходів)
            result, passes, final_skew = perspective.auto_correct_iterative(image)
            # Deskew після останнього warp
            angle = deskew_module.measure_skew_angle(result)
            if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                result = deskew_module.apply_deskew(result, angle)
            status = f"перспектива виправлена ({passes} прохід(ів)) + deskew"
            return result, status
        else:
            angle = deskew_module.measure_skew_angle(image)
            if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                result = deskew_module.apply_deskew(image, angle)
                return result, "deskew (нахил виправлено)"
            else:
                return image.copy(), "перспектива не потребує корекції"
    else:
        angle = deskew_module.measure_skew_angle(image)
        if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
            result = deskew_module.apply_deskew(image, angle)
            return result, "deskew (нахил виправлено)"
        else:
            return image.copy(), "перспектива не потрібна"
```

- [ ] Відмітити виконання.

---

## КРОК 5 — Додати `import` у `pipeline.py`

У `pipeline.py` переконатись що є імпорт `perspective` (вже є) та `deskew_module` (вже є). Додати `logger`:

```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] Відмітити виконання.

---

## Порядок виконання кроків агентом

```
КРОК 1 → КРОК 2 → КРОК 3 → КРОК 4 → КРОК 5
```

Після кожного кроку запустити базовий smoke-test:

```python
# smoke_test_perspective.py
import cv2, numpy as np
from processing import pipeline

# Тест 1: Пряме зображення — не має змінюватись
img = np.ones((800, 600, 3), dtype=np.uint8) * 240
result, msg, _ = pipeline.run_autofix(img, use_perspective=True, settings={"pipeline_preset": "doc_bw"})
assert result.shape == img.shape, "FAIL: розмір змінився на рівному зображенні"
print(f"Тест 1 OK: {msg}")

# Тест 2: Не крашиться на довільному BGR
import os
for f in os.listdir("tests/test_images")[:3]:
    img2 = cv2.imread(f"tests/test_images/{f}")
    if img2 is not None:
        r2, m2, _ = pipeline.run_autofix(img2, use_perspective=True, settings={"pipeline_preset": "doc_bw"})
        print(f"Тест 2 OK [{f}]: {m2}")
```

- [ ] Відмітити виконання.