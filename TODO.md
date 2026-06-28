## 📋 План для агента кодування — PhotoPrint: виправлення детекції перспективи та кольорового відтінку

---

## 🎯 Загальна мета
Виправити три незалежні проблеми:
- **Проблема 1:** Авто-перспектива не знаходить документ і пише "перспектива не потрібна"
- **Проблема 2:** Ручна перспектива виставляє точки "де попало"
- **Проблема 3:** Кольоровий відтінок фону (жовтий/теплий) не видаляється

---

## ⚠️ Правила для агента
- Не переписувати модулі повністю — тільки точкові зміни
- Після кожного кроку — коміт або збереження з коментарем
- Не чіпати `batch_processor.py`, `printer.py`, `saver.py`, `loader.py`
- Всі нові константи виносити у верх файлу поруч з існуючими

---

## БЛОК 1 — Виправлення авто-перспективи

### Крок 1.1 — Підняти ліміт площі в `_score_quad()`
**Файл:** `processing/perspective.py`

**Проблема:** Документи що займають >92% кадру отримують score=0.0 і відкидаються. Реальні телефонні фото — документ займає 85–97% кадру.

**Що змінити:**
- Знайти константу `MAX_QUAD_AREA_RATIO = 0.97` — вона вже є у файлі
- Знайти функцію `_score_quad()` — там жорстко прописано `0.92` у тілі функції (не через константу)
- Замінити `0.92` на `MAX_QUAD_AREA_RATIO` щоб використовувалась константа
- Змінити пік скору: зараз `abs(ratio - 0.50) / 0.42` — замінити на `abs(ratio - 0.60) / 0.37` (пік на 60% а не 50%, бо телефонні фото документів займають більше кадру)

- [ ] Відмітити виконання

---

### Крок 1.2 — Знизити поріг детекції викривлення в `detect_skewed_sides()`
**Файл:** `processing/perspective.py`

**Проблема:** `PARTIAL_SKEW_THRESHOLD_RATIO = 0.03` (3%). При розмірі 800px поріг = 24px. Реальне відхилення на телефонних фото = 10–22px → система вважає перспективу ідеальною.

**Що змінити:**
- Знайти `PARTIAL_SKEW_THRESHOLD_RATIO = 0.03`
- Замінити на `PARTIAL_SKEW_THRESHOLD_RATIO = 0.015`
- Додати коментар: `# знижено з 0.03: телефонні фото мають невелике але реальне викривлення`

- [ ] Відмітити виконання

---

### Крок 1.3 — Розширити фільтр ліній в `_try_hough_lines()` для повернутих зображень
**Файл:** `processing/perspective.py`

**Проблема:** Зображення 0(14) повернуте на 90°. Лінії що мали б бути "горизонтальними" стають вертикальними. `HOUGH_ANGLE_TOLERANCE_DEG = 15°` — занадто вузький допуск.

**Що змінити:**
- Знайти `HOUGH_ANGLE_TOLERANCE_DEG = 15.0`
- Замінити на `HOUGH_ANGLE_TOLERANCE_DEG = 20.0`
- У функції `_try_hough_lines()` знайти умову класифікації ліній:
```python
if angle_deg < HOUGH_ANGLE_TOLERANCE_DEG:  # горизонтальна
elif angle_deg > (90.0 - HOUGH_ANGLE_TOLERANCE_DEG):  # вертикальна
```
- Ця умова вже використовує константу — змінювати нічого додатково не треба, достатньо зміни константи

- [ ] Відмітити виконання

---

### Крок 1.4 — Пом'якшити фільтр відступу від краю кадру в `_find_quad_contour()`
**Файл:** `processing/perspective.py`

**Проблема:** `_touches_image_border()` з `BORDER_MARGIN_PX = 2` відкидає контури що торкаються краю. Але для документів 0(5) і 0(14) — документ торкається або майже торкається країв кадру.

**Що змінити:**
- У функції `_find_quad_contour()` знайти виклик:
```python
if _touches_image_border(cnt, gray_shape) and area > image_area * MAX_BORDER_AREA_RATIO:
    continue
```
- Змінити умову: замість `MAX_BORDER_AREA_RATIO` (0.92) використовувати жорсткіший поріг `0.98` саме тут:
```python
if _touches_image_border(cnt, gray_shape) and area > image_area * 0.98:
    continue
```
- Коментар: `# пом'якшено: документ може торкатись країв кадру`

- [ ] Відмітити виконання

---

### Крок 1.5 — Додати fallback через MORPH_GRADIENT у `_detect_corners_impl()`
**Файл:** `processing/perspective.py`

**Проблема:** Для гільошованих документів (орнамент) adaptive threshold і Canny знаходять безліч внутрішніх контурів. Потрібен метод що ігнорує внутрішню текстуру і шукає тільки зовнішній контур.

**Що змінити:**
- Додати нову приватну функцію `_try_external_contour(gray)` після функції `_try_hough_lines()`:

```python
def _try_external_contour(gray: np.ndarray) -> np.ndarray | None:
    """
    Fallback для документів з багатою внутрішньою текстурою (гільош, орнамент).
    Використовує сильне розмиття щоб прибрати внутрішню текстуру,
    залишаючи тільки зовнішні межі документа.
    """
    # Сильне розмиття прибирає внутрішній орнамент
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    # Otsu threshold шукає глобальний розподіл (документ vs фон)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Морфологія для закриття дрібних прогалин
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_contour(closed, gray.shape)
```

- У `_detect_corners_impl()` додати цей метод в список `methods` останнім:
```python
("external_contour", lambda g: _try_external_contour(g)),
```

- [ ] Відмітити виконання

---

### Крок 1.6 — Додати попередню ротацію для детекції при повернутому зображенні
**Файл:** `processing/perspective.py`

**Проблема:** Зображення 0(14) повернуте на 90°. Детектор не адаптується.

**Що змінити:**
- Додати нову функцію `_detect_with_rotation_candidates(gray)` після `_detect_corners_impl()`:

```python
def _detect_with_rotation_candidates(gray: np.ndarray) -> np.ndarray | None:
    """
    Пробує детекцію для 4 орієнтацій зображення (0°, 90°, 180°, 270°).
    Якщо знаходить кути — перераховує їх координати назад у простір оригіналу.
    Повертає найкращий результат або None.
    """
    h, w = gray.shape[:2]
    best_score = 0.0
    best_corners = None

    rotations = [
        (0,   gray),
        (90,  cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(gray, cv2.ROTATE_180)),
        (270, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    for angle, rotated in rotations:
        corners = _detect_corners_impl(rotated)
        if corners is None:
            continue
        rh, rw = rotated.shape[:2]
        score = _score_quad(corners, rotated.shape)
        if score <= best_score:
            continue
        # Перераховуємо координати назад у простір оригінального gray
        if angle == 0:
            back = corners
        elif angle == 90:
            # (x, y) у rotated → (y, w - x) у оригіналі (де w = h оригіналу)
            back = np.array([[c[1], w - c[0]] for c in corners], dtype=np.float32)
        elif angle == 180:
            back = np.array([[rw - c[0], rh - c[1]] for c in corners], dtype=np.float32)
        elif angle == 270:
            # (x, y) у rotated → (h - y, x) у оригіналі
            back = np.array([[h - c[1], c[0]] for c in corners], dtype=np.float32)
        best_score = score
        best_corners = back

    return best_corners
```

- У `auto_detect_corners()` замінити виклик `_detect_corners_impl(gray)` на `_detect_with_rotation_candidates(gray)`

- [ ] Відмітити виконання

---

## БЛОК 2 — Виправлення ручної перспективи

### Крок 2.1 — Покращити fallback точок у `_do_persp_manual()`
**Файл:** `gui/main_window.py`

**Проблема:** Коли `detect_corners()` повертає `None`, виставляються точки 80% центру зображення незалежно від того де реально знаходиться документ. Для 0(14) — документ у нижній частині кадру, точки — посередині.

**Що змінити:**
- Знайти функцію `_default_perspective_corners(self, image)` у `main_window.py`
- Вона повертає `margin = 10%`. Це залишити.
- Знайти у `_do_persp_manual()` рядок:
```python
if corners is None:
    corners = self._default_perspective_corners(self._base_for_perspective)
    status = "Встановіть кути вручну (документ не знайдено)"
```
- Перед `_default_perspective_corners` додати спробу через агресивніший пошук:
```python
if corners is None:
    # Спроба 2: з пониженим порогом площі через _try_external_contour
    from processing.perspective import _try_external_contour, _refine_corners_subpix, _order_points
    import cv2 as _cv2
    _gray = _cv2.cvtColor(self._base_for_perspective, _cv2.COLOR_BGR2GRAY)
    _small_scale = min(800 / max(_gray.shape[:2]), 1.0)
    _small = _cv2.resize(_gray, None, fx=_small_scale, fy=_small_scale) if _small_scale < 1.0 else _gray
    corners = _try_external_contour(_small)
    if corners is not None:
        corners = (corners / _small_scale).astype(np.float32)
        corners = _refine_corners_subpix(_gray, corners)
        status = "Тягніть кути для корекції перспективи (знайдено резервним методом)"
    else:
        corners = self._default_perspective_corners(self._base_for_perspective)
        status = "Встановіть кути вручну (документ не знайдено)"
```

- [ ] Відмітити виконання

---

### Крок 2.2 — Зробити точки перспективи більшими і видимішими
**Файл:** `gui/preview.py`

**Проблема:** `POINT_RADIUS = 9` — на прев'ю зображенні 900px це дуже дрібно, важко потрапити.

**Що змінити:**
- Знайти `POINT_RADIUS = 9`
- Замінити на `POINT_RADIUS = 12`
- Знайти `POINT_HIT_RADIUS_MULTIPLIER = 2`
- Замінити на `POINT_HIT_RADIUS_MULTIPLIER = 3`
- Знайти `POINT_HIT_TOLERANCE = 4`
- Замінити на `POINT_HIT_TOLERANCE = 8`

- [ ] Відмітити виконання

---

### Крок 2.3 — Додати підказку при активації ручної перспективи
**Файл:** `gui/main_window.py`

**Що змінити:**
- Знайти у `_do_persp_manual()` де встановлюється статус
- Якщо статус = "Тягніть кути для корекції перспективи" — додати після нього:
```
" | Перетягуйте кольорові точки (TL=червона, TR=зелена, BR=синя, BL=жовта)"
```

- [ ] Відмітити виконання

---

## БЛОК 3 — Виправлення кольорового відтінку

### Крок 3.1 — Створити новий модуль `processing/color_cast.py`
**Файл:** `processing/color_cast.py` (новий файл)

**Проблема:** Жовтий/теплий відтінок від телефонної камери не видаляється тому що `shadow_remove` працює тільки з L-каналом LAB, а відтінок — це зміщення у каналах a та b.

**Що створити — новий модуль з функцією `correct_color_cast()`:**

```python
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
    # Плавний перехід від 0 (темне) до 1 (світле) через 150-200 L
    weight = np.clip((l_ch.astype(np.float32) - 140.0) / 60.0, 0.0, 1.0)

    a_corrected = a_ch + a_shift * weight * COLOR_CAST_BLEND
    b_corrected = b_ch + b_shift * weight * COLOR_CAST_BLEND

    a_result = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_result = np.clip(b_corrected, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_ch, a_result, b_result])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result, True
```

- [ ] Відмітити виконання

---

### Крок 3.2 — Підключити `color_cast` у `pipeline.py`
**Файл:** `processing/pipeline.py`

**Що змінити:**

- Знайти рядок імпортів на початку файлу:
```python
from processing import autofix, sharpen, hdr, perspective, brightness_contrast as bc, ...
```
- Додати в цей рядок: `color_cast`

- У функції `run_autofix()` знайти цикл `for step_key, _ in PIPELINE_STEPS_FIXED_ORDER:`
- Знайти блок `elif step_key == "shadow_remove":` — після всього блоку shadow_remove (після `shadow_highlight`) додати новий блок:

```python
        # --- Корекція кольорового відтінку фону ---
        # Виконується після shadow_remove, бо тіні можуть спотворювати аналіз кольору
        if doc_type in (DocType.COLOR_DOCUMENT.value, DocType.BW_DOCUMENT.value):
            result, had_cast = color_cast.correct_color_cast(result)
            if had_cast:
                log_entries.append({"step": "color_cast", "applied": True,
                                    "detail": "відтінок нейтралізовано"})
```

- [ ] Відмітити виконання

---

### Крок 3.3 — Додати `color_cast` у список кроків `PIPELINE_STEPS_FIXED_ORDER`
**Файл:** `processing/pipeline.py`

**Що змінити:**
- Знайти `PIPELINE_STEPS_FIXED_ORDER` у `pipeline.py`
- Додати рядок після `("shadow_remove", ...)`:
```python
("color_cast", "Нейтралізація відтінку"),
```
- Також додати `("color_cast", ...)` у `PIPELINE_STEPS_FIXED_ORDER` у `gui/settings_window.py` у список `PIPELINE_STEPS_FIXED_ORDER`

- [ ] Відмітити виконання

---

### Крок 3.4 — Посилити існуючу нейтралізацію для `phone_camera`
**Файл:** `processing/pipeline.py`

**Проблема:** Поточний код нейтралізує лише 30% відтінку після shadow_remove.

**Що змінити:**
- Знайти у `run_autofix()`:
```python
a_shift = (128.0 - float(np.median(a_bg))) * 0.3
b_shift = (128.0 - float(np.median(b_bg))) * 0.3
```
- Замінити `0.3` на `0.7` в обох рядках
- Коментар: `# посилено з 0.3 до 0.7: відтінок телефонної камери потребує сильнішої корекції`

- [ ] Відмітити виконання

---

## БЛОК 4 — Фінальна перевірка та інтеграція

### Крок 4.1 — Додати `color_cast` у `processing/__init__.py`
**Файл:** `processing/__init__.py`

**Що змінити:**
- Файл зараз порожній — залишити порожнім (імпорти йдуть напряму через `from processing import ...`)
- Перевірити що `processing/color_cast.py` знаходиться у правильній директорії

- [ ] Відмітити виконання

---

### Крок 4.2 — Перевірити константи у `app_settings.py`
**Файл:** `config/app_settings.py`

**Що змінити:**
- Нових налаштувань для `color_cast` у settings.ini не потрібно — модуль автоматичний
- Перевірити що `DEFAULT_PIPELINE_STEPS_ENABLED` містить нові кроки якщо потрібно

- [ ] Відмітити виконання

---

### Крок 4.3 — Ручне тестування на файлах 0(5) та 0(14)
**Дія (не код):**

Після всіх змін запустити програму та перевірити:

**Тест 1 — файл 0(5) (медичний документ):**
- [ ] Авто-перспектива знаходить документ (не пише "не потрібна")
- [ ] Ручна перспектива — точки на кутах документа, не посередині кадру
- [ ] Жовтого відтінку немає або він значно послаблений

**Тест 2 — файл 0(14) (військовий квиток, повернутий на 90°):**
- [ ] Авто-перспектива спрацьовує
- [ ] Ручна перспектива — точки в адекватних місцях
- [ ] Кольоровий фон з гільошем — відтінок нейтралізований

- [ ] Відмітити виконання

---

## 📊 Підсумок змін

| Блок | Файли | Тип змін |
|------|-------|----------|
| Авто-перспектива | `perspective.py` | 6 точкових змін + 2 нові функції |
| Ручна перспектива | `main_window.py`, `preview.py` | 3 точкові зміни |
| Кольоровий відтінок | `color_cast.py` (новий), `pipeline.py`, `settings_window.py` | 1 новий файл + 3 точкові зміни |

**Загальна кількість файлів:** 5 існуючих + 1 новий