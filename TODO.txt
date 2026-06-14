# Фінальний план реалізації адаптивного Full Auto pipeline для PhotoPrint

## Принципи реалізації

Перед деталями — кілька незмінних правил для агента:

Ніколи не видаляти і не змінювати існуючий `run_autofix` — він залишається стабільним режимом. `run_full_auto` — це окремий оркестратор поверх вже наявних примітивів.

Діагностичні метрики повертають не тільки `bool` але й числове значення сили — це множник, а не перемикач. Умова "застосувати чи ні" замінюється на "застосувати з силою X", де X може бути дуже малим числом що практично еквівалентно "не застосовувати".

Після кожного кроку що суттєво змінює гістограму або геометрію — частково перераховувати діагностику перед наступним кроком.

Реалізовувати поетапно: спочатку тільки діагностика і логування, потім поступово додавати корекції в порядку зростання ризику артефактів.

---

## Фаза А — Модуль діагностики (нульовий ризик)

### Файл `processing/diagnostics.py` — створити новий

#### Константи на початку файлу

```python
DIAGNOSTICS_RESIZE_MAX = 600       # розмір для швидкої діагностики

GRADIENT_L_DIFF_THRESHOLD = 30     # мінімальна різниця медіан для "є градієнт"
GRADIENT_L_MAX_EXPECTED = 120      # "максимальний" очікуваний діапазон (для нормалізації)

CONTRAST_RANGE_THRESHOLD = 160     # якщо p99-p1 < цього — контраст низький
CONTRAST_RANGE_MAX = 254           # максимально можливий діапазон

BLUR_THRESHOLD = 80.0              # збіг з autosharp_threshold в налаштуваннях
BLUR_MAX_VARIANCE = 500.0          # вище цього — "достатньо різке"

DARK_MEAN_THRESHOLD = 80           # середнє L нижче цього — "занадто темне"
BRIGHT_MEAN_THRESHOLD = 200        # середнє L вище цього — "занадто світле"
OVEREXPOSED_L_THRESHOLD = 250      # пікселі вище цього вважаються пересвіченими
UNDEREXPOSED_L_THRESHOLD = 5       # пікселі нижче цього вважаються недосвіченими

PERSPECTIVE_SKEW_THRESHOLD = 0.03  # 3% від розміру зображення — мінімальне викривлення
```

#### Клас `DiagnosticResult`

Реалізувати як `@dataclass`. Поля:

```
doc_type: str                    # "bw_document" / "color_document" / "photo"

# Градієнт фону
gradient_has: bool
gradient_strength: float         # 0..1, нормалізована сила
gradient_direction: str          # "horizontal" / "vertical" / "both" / "none"

# Контраст
contrast_range_l: float          # p99 - p1 каналу L
contrast_strength_needed: float  # 0..1, наскільки потрібна корекція
overexposed_ratio: float         # частка пікселів > 250
underexposed_ratio: float        # частка пікселів < 5

# Яскравість
brightness_mean_l: float         # середнє L
brightness_correction: float     # -1..1, відхилення від нейтрального

# Розмиття
blur_variance: float             # Laplacian variance
blur_strength_needed: float      # 0..1, наскільки потрібна різкість
blur_sharpen_strength: float     # конкретне значення сили різкості 0..1

# Перспектива
perspective_has: bool
perspective_corners: np.ndarray | None   # shape (4,2) або None
perspective_skew_ratio: float    # відносне відхилення від прямокутника
```

#### Функція `_resize_for_analysis(image, max_dim) -> np.ndarray`

Внутрішня (з підкресленням). Зменшує зображення до `max_dim` по більшій стороні якщо більше. Повертає копію або оригінал якщо вже менше. Використовується усіма діагностичними функціями крім `_measure_perspective`.

#### Функція `_measure_gradient(image) -> tuple[bool, float, str]`

Алгоритм:
- Взяти L-канал (LAB).
- Розбити на сітку 4×4 блоки (16 блоків).
- Для кожного блоку обчислити медіану L.
- Побудувати матрицю медіан 4×4.
- `diff_total = max_медіана - min_медіана`.
- `std_rows = np.std(np.mean(matrix, axis=1))` — варіація по рядках (горизонтальний градієнт).
- `std_cols = np.std(np.mean(matrix, axis=0))` — варіація по стовпцях (вертикальний градієнт).
- `has_gradient = diff_total > GRADIENT_L_DIFF_THRESHOLD`.
- `strength = min(1.0, diff_total / GRADIENT_L_MAX_EXPECTED)`.
- `direction`: якщо `std_cols > std_rows * 1.5` → `"horizontal"`, якщо `std_rows > std_cols * 1.5` → `"vertical"`, якщо обидва значні → `"both"`, інакше → `"none"`.
- Повернути `(has_gradient, strength, direction)`.

#### Функція `_measure_contrast(image) -> tuple[float, float, float, float]`

Алгоритм:
- Взяти L-канал.
- `p1 = np.percentile(L, 1)`, `p99 = np.percentile(L, 99)`.
- `range_l = p99 - p1`.
- `strength_needed = max(0.0, (CONTRAST_RANGE_THRESHOLD - range_l) / CONTRAST_RANGE_THRESHOLD)` — якщо діапазон широкий то `strength_needed = 0`.
- `overexposed_ratio = float(np.mean(L > OVEREXPOSED_L_THRESHOLD))`.
- `underexposed_ratio = float(np.mean(L < UNDEREXPOSED_L_THRESHOLD))`.
- Повернути `(range_l, strength_needed, overexposed_ratio, underexposed_ratio)`.

#### Функція `_measure_brightness(image) -> tuple[float, float]`

Алгоритм:
- Взяти L-канал.
- `mean_l = float(np.mean(L))`.
- `correction = (127.0 - mean_l) / 127.0` — позитивне якщо темне, негативне якщо світле.
- Обмежити `correction` в `[-1.0, 1.0]`.
- Повернути `(mean_l, correction)`.

#### Функція `_measure_blur(image, settings) -> tuple[float, float, float]`

Алгоритм:
- Викликати існуючий `sharpen.measure_sharpness(image)` → `variance`.
- `max_sharpen = settings.get("autosharp_max_strength", 0.7)`.
- `threshold = settings.get("autosharp_threshold", BLUR_THRESHOLD)`.
- Якщо `variance >= threshold` → `strength_needed = 0.0`, `sharpen_strength = 0.0`.
- Інакше: `strength_needed = max(0.0, 1.0 - variance / threshold)`, `sharpen_strength = max(0.15, strength_needed * max_sharpen)`.
- Повернути `(variance, strength_needed, sharpen_strength)`.

#### Функція `_measure_perspective(image) -> tuple[bool, np.ndarray | None, float]`

Алгоритм:
- Викликати `perspective.auto_detect_corners(image)` → `corners`.
- Якщо `corners is None` → повернути `(False, None, 0.0)`.
- `ordered = perspective._order_points(corners)`.
- `tl, tr, br, bl = ordered`.
- `top_skew = abs(tl[1] - tr[1])`, `bottom_skew = abs(bl[1] - br[1])`.
- `left_skew = abs(tl[0] - bl[0])`, `right_skew = abs(tr[0] - br[0])`.
- `max_skew = max(top_skew, bottom_skew, left_skew, right_skew)`.
- `h, w = image.shape[:2]`.
- `skew_ratio = max_skew / max(h, w)`.
- `has_perspective = skew_ratio > PERSPECTIVE_SKEW_THRESHOLD`.
- Повернути `(has_perspective, corners if has_perspective else None, skew_ratio)`.

**Важливо:** ця функція не викликає `_resize_for_analysis` бо `auto_detect_corners` вже має власний `max_dim` параметр.

#### Функція `_measure_doc_type(image, settings) -> str`

Просто викликає існуючий `doc_classifier.classify(image, bw_std_thresh=settings.get(...), ...)`. Повертає рядок.

#### Головна функція `diagnose(image: np.ndarray, settings: dict) -> DiagnosticResult`

Алгоритм:
- `small = _resize_for_analysis(image, DIAGNOSTICS_RESIZE_MAX)`.
- Послідовно викликати всі `_measure_*` функції на `small` (крім `_measure_perspective` — викликати на оригіналі `image`).
- Зібрати `DiagnosticResult` з усіх результатів.
- Повернути.

**Вхідне зображення не змінювати ніде в цьому файлі.**

#### Функція `partial_rediagnose(image: np.ndarray, settings: dict, fields: list[str]) -> dict`

Для часткового перерахунку після проміжних кроків. Приймає список полів для перерахунку (`["contrast", "brightness", "blur"]`), повертає словник з оновленими значеннями. Не перераховує перспективу і тип документа (вони не змінюються після тональних корекцій).

---

## Фаза Б — Низький ризик: різкість і яскравість

### Файл `processing/pipeline.py` — додати функцію `run_full_auto`

Повна сигнатура:

```python
def run_full_auto(
    image: np.ndarray,
    settings: dict,
    dry_run: bool = False,
) -> tuple[np.ndarray, str, dict]
```

Параметр `dry_run=True` — тільки діагностика, нічого не застосовувати, повернути `(image.copy(), status, applied_steps)`. Це для Фази А логування.

Повертає: `(result, status_message, applied_steps)` де `applied_steps: dict[str, float | bool]` — що було застосовано і з якою силою.

#### Внутрішня структура `run_full_auto`

**Ініціалізація:**
```
result = image.copy()
applied_steps = {}
status_parts = []
```

**Крок 0 — Діагностика:**
```
from processing import diagnostics
diag = diagnostics.diagnose(image, settings)
```
Якщо `dry_run`: зібрати статус з діагностики і одразу повернути.

**Крок 1 — Видалення градієнтного фону:**

Умова застосування:
- `diag.gradient_has is True`.
- `diag.doc_type != "photo"` (для фото — пропустити повністю на старті).
- `diag.gradient_strength > 0.3` (тільки якщо градієнт достатньо сильний).

Якщо умова виконана:
- `result, had_shadow = shadow_remove.auto_remove_shadow(result)`.
- Якщо `had_shadow`: `applied_steps["shadow_remove"] = diag.gradient_strength`.

Після цього кроку — частковий перерахунок:
- `updated = diagnostics.partial_rediagnose(result, settings, ["contrast", "brightness"])`.
- Оновити `diag.contrast_range_l`, `diag.contrast_strength_needed`, `diag.brightness_mean_l`, `diag.brightness_correction` з `updated`.

**Крок 2 — Корекція перспективи:**

Умова: `diag.perspective_has is True`.

Якщо умова:
- `result = perspective.apply_correction(result, diag.perspective_corners)`.
- `applied_steps["perspective"] = diag.perspective_skew_ratio`.

Після цього кроку — частковий перерахунок:
- `updated = diagnostics.partial_rediagnose(result, settings, ["contrast", "brightness", "blur"])`.
- Оновити відповідні поля `diag`.

**Крок 3 — Яскравість:**

Не бінарне рішення — завжди обчислювати силу:
- `brightness_strength = abs(diag.brightness_correction)`.
- Мінімальний поріг застосування: `if brightness_strength < 0.05: пропустити`.
- Якщо `diag.brightness_correction > 0` (темне): `result = bc.auto_brightness(result, percentile_low=2.0, percentile_high=98.0)`.
- Якщо `diag.brightness_correction < 0` (світле): `result = bc.auto_brightness(result, percentile_low=5.0, percentile_high=95.0)`.
- `applied_steps["brightness"] = brightness_strength`.

Після — частковий перерахунок контрасту:
- `updated = diagnostics.partial_rediagnose(result, settings, ["contrast"])`.
- Оновити `diag.contrast_strength_needed`.

**Крок 4 — Контраст:**

- `contrast_strength = diag.contrast_strength_needed`.
- Мінімальний поріг: `if contrast_strength < 0.05: пропустити`.
- Обмеження зверху: `contrast_strength = min(contrast_strength, 0.85)` — не перестаратись.
- `contrast_mode = settings.get("contrast_mode", "linear")`.
- `result = run_contrast_advanced(result, contrast_strength, contrast_mode)`.
- `applied_steps["contrast"] = contrast_strength`.

Після — частковий перерахунок розмиття:
- `updated = diagnostics.partial_rediagnose(result, settings, ["blur"])`.
- Оновити `diag.blur_sharpen_strength`.

**Крок 5 — HDR (тільки для фото):**

Умова: `diag.doc_type == "photo" AND settings.get("hdr_in_autofix", True)`.
- `hdr_strength = settings.get("hdr_strength", 0.5)`.
- `result = hdr.apply_adaptive(result, strength=hdr_strength)`.
- `applied_steps["hdr"] = hdr_strength`.

**Крок 6 — Специфічна обробка по типу документа:**

- `sharpen_strength = diag.blur_sharpen_strength` (або `settings.get("sharpen_strength", 0.4)` якщо `blur_sharpen_strength == 0`).
- `doc_type == "bw_document"`: `result = autofix.apply_bw_document(result, sharpen_strength=sharpen_strength, binary=settings.get("bw_binary", False))`.
- `doc_type == "color_document"`: `result = autofix.apply_color_document(result, sharpen_strength=sharpen_strength)`.
- `doc_type == "photo"`: тільки `result = sharpen.apply(result, strength=sharpen_strength)` якщо `diag.blur_strength_needed > 0.05`.
- `applied_steps["doc_processing"] = diag.doc_type`.

**Крок 7 — Shadow highlight:**

- `sh_strength = settings.get("shadow_highlight_strength", 0.0)`.
- Якщо `sh_strength > 0.001`: `result = shadow_highlight.apply_shadow_highlight(result, strength=sh_strength)`.

**Крок 8 — Додатковий контраст Auto Fix:**

- `autofix_contrast = settings.get("autofix_contrast", 0.15)`.
- Якщо `autofix_contrast > 0.001`: `result = run_contrast_advanced(result, autofix_contrast, contrast_mode)`.

**Крок 9 — Формат виходу:**

Аналогічно існуючому `run_autofix`:
- `output_color_mode = settings.get("output_color_mode", "auto")`.
- Якщо `grayscale`: `result = bc.to_grayscale(result)`.
- Якщо `binary`: конвертація через `adaptiveThreshold`.
- `auto` — залишити як є (визначено типом документа).

**Збирання статусу:**
- Для кожного ключа в `applied_steps` додати рядок у `status_parts`.
- `status_msg = "Full Auto: " + ", ".join(status_parts)`.
- Повернути `(result, status_msg, applied_steps)`.

---

## Фаза В — Нові функції контрасту

### Файл `processing/brightness_contrast.py` — додати три функції

Додати в кінець файлу після існуючих функцій. Не змінювати існуючі.

#### Нові константи

```python
PERCENTILE_CONTRAST_MIN_RANGE = 20   # якщо діапазон вже широкий — нічого не робити
S_CURVE_GAMMA_BASE = 1.0
S_CURVE_GAMMA_MULTIPLIER = 2.0       # gamma = 1.0 + strength * 2.0
```

#### Функція `smart_contrast_percentile(image, strength, low_perc=1.0, high_perc=99.0)`

Алгоритм:
- Взяти L-канал (LAB).
- `p_low = np.percentile(L, low_perc)`, `p_high = np.percentile(L, high_perc)`.
- Якщо `p_high - p_low < PERCENTILE_CONTRAST_MIN_RANGE` → повернути `image.copy()`.
- `l_float = L.astype(np.float32)`.
- `l_stretched = np.clip((l_float - p_low) / (p_high - p_low) * 255.0, 0, 255)`.
- `l_result = np.clip(l_float * (1.0 - strength) + l_stretched * strength, 0, 255).astype(np.uint8)`.
- Зібрати LAB, конвертувати в BGR, повернути.

#### Функція `contrast_s_curve(image, strength)`

Алгоритм:
- Взяти L-канал (LAB).
- `l_norm = L.astype(np.float32) / 255.0`.
- `gamma = S_CURVE_GAMMA_BASE + strength * S_CURVE_GAMMA_MULTIPLIER`.
- `x_g = np.power(np.clip(l_norm, 0, 1), gamma)`.
- `inv_g = np.power(np.clip(1.0 - l_norm, 0, 1), gamma)`.
- `s = x_g / (x_g + inv_g + 1e-6)`.
- `l_result = np.clip(s * 255.0, 0, 255).astype(np.uint8)`.
- Зібрати LAB, конвертувати в BGR, повернути.

#### Функція `local_contrast_adaptive(image, strength)`

Алгоритм:
- Взяти L-канал (LAB).
- `mask = detail_map.detail_mask(L)` — з `processing.detail_map` (вже існує).
- Обчислити S-криву для всього L (та сама логіка що вище) → `l_contrasted`.
- `l_orig = L.astype(np.float32)`.
- `l_result = np.clip(l_orig * (1.0 - mask) + l_contrasted * mask, 0, 255).astype(np.uint8)`.
- Зібрати LAB, конвертувати в BGR, повернути.

**Важливо для всіх трьох функцій:** вхідне зображення не змінювати. Повертати новий масив.

---

## Фаза Г — Оновлення конфігурації

### Файл `config/app_settings.py`

**Нові константи:**
```python
DEFAULT_FULL_AUTO_MODE = False
DEFAULT_FULL_AUTO_MIN_GRADIENT_STRENGTH = 0.3   # мінімальна сила градієнту для shadow_remove
```

**У `load()`** додати:
```python
"full_auto_mode": cfg.getboolean("processing", "full_auto_mode", fallback=DEFAULT_FULL_AUTO_MODE),
"full_auto_min_gradient_strength": cfg.getfloat("processing", "full_auto_min_gradient_strength", fallback=DEFAULT_FULL_AUTO_MIN_GRADIENT_STRENGTH),
```

**У `save()`** додати відповідні записи в секцію `[processing]`.

---

## Фаза Д — Оновлення GUI

### Файл `gui/settings_window.py`

**Нова секція** "Адаптивний режим (Full Auto)". Розмістити в правій колонці після секції "Режим запуску".

Елементи секції:
- `QCheckBox("Використовувати Full Auto замість Auto Fix")` → `self._cb_full_auto_mode`.
- `QLabel` з поясненням (дрібний шрифт, сірий): `"Full Auto аналізує кожне зображення окремо і застосовує лише потрібні корекції з адаптивною силою."`.

**Оновити `_apply_settings`:**
```python
self._cb_full_auto_mode.setChecked(s.get("full_auto_mode", False))
```

**Оновити `_collect_settings`:**
```python
"full_auto_mode": self._cb_full_auto_mode.isChecked(),
```

**Вже наявний `_combo_contrast_mode`** — нічого не змінювати, він вже в коді.

### Файл `gui/main_window.py`

**Перейменувати** тіло існуючого `_do_autofix` на `_do_autofix_classic` (приватний метод).

**Новий `_do_autofix`:**

```python
def _do_autofix(self):
    if self._settings.get("full_auto_mode", False):
        self._do_full_auto()
    else:
        self._do_autofix_classic()
```

**Новий `_do_full_auto`:**

```python
def _do_full_auto(self):
    if self._orig is None:
        self._set_status("Спочатку оберіть файл")
        return
    try:
        result, status_msg, applied_steps = pipeline.run_full_auto(
            self._base,
            settings=self._settings,
        )
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
        self._preview.set_autofix_applied(True)
        self._set_status(status_msg)
        self._logger.debug(f"Full Auto applied_steps: {applied_steps}")
        self._update_buttons()
    except Exception as e:
        self._logger.error(f"Помилка Full Auto: {e}", exc_info=True)
        self._set_status(f"Помилка Full Auto: {e}")
```

**Оновити `_on_settings_saved`** — після збереження оновити текст кнопки `btn_autofix`:
- Якщо `s.get("full_auto_mode", False)` → `self._btn_autofix.setText("⚡ Full Auto")`.
- Інакше → `self._btn_autofix.setText("⚡ Auto Fix")`.

**Оновити `_apply_default_mode`** — аналогічно встановити текст кнопки при старті.

### Файл `batch/batch_processor.py`

**У методі `run_auto`** всередині циклу замінити блок автофікс:

```python
if s.get("autofix_enabled", True):
    if s.get("full_auto_mode", False):
        processed, _, _ = pipeline.run_full_auto(image, settings=s)
    else:
        processed, _ = pipeline.run_autofix(image, ...)
else:
    processed = image
```

Параметри для `run_autofix` залишити точно такими як є зараз.

---

## Фаза Е — Тести

### Файл `tests/unit/test_diagnostics.py` — новий файл

#### Клас `TestMeasureGradient`

- `test_flat_image_no_gradient` — однотонне L=128 → `gradient_has = False`, `strength ≈ 0`.
- `test_strong_horizontal_gradient` — L від 50 (ліво) до 220 (право) → `gradient_has = True`, `direction = "horizontal"`, `strength > 0.7`.
- `test_strong_vertical_gradient` — L від 50 (верх) до 220 (низ) → `direction = "vertical"`.
- `test_weak_gradient_below_threshold` — різниця медіан 20 (< 30) → `gradient_has = False`.
- `test_gradient_immutable` — вхідне зображення не змінюється.

#### Клас `TestMeasureContrast`

- `test_narrow_histogram` — всі пікселі L в [100, 130] → `strength_needed > 0.7`.
- `test_wide_histogram` — L від 0 до 255 → `strength_needed = 0.0`.
- `test_overexposed_ratio` — 40% пікселів L > 250 → `overexposed_ratio ≈ 0.4`.
- `test_underexposed_ratio` — 30% пікселів L < 5 → `underexposed_ratio ≈ 0.3`.
- `test_strength_proportional_to_narrowness` — вужча гістограма → більший `strength_needed`.

#### Клас `TestMeasureBlur`

- `test_sharp_noise_image` — випадковий шум → `blur_strength_needed = 0.0`.
- `test_gaussian_blurred` — після `GaussianBlur(21,21)` → `blur_strength_needed > 0.5`.
- `test_sharpen_strength_bounded` — `blur_sharpen_strength <= max_sharpen`.
- `test_blur_immutable` — вхідне зображення не змінюється.

#### Клас `TestMeasureBrightness`

- `test_dark_image` — L ≈ 50 → `brightness_correction > 0.5`.
- `test_bright_image` — L ≈ 220 → `brightness_correction < -0.3`.
- `test_neutral_image` — L ≈ 127 → `abs(brightness_correction) < 0.05`.

#### Клас `TestMeasurePerspective`

- `test_straight_document` — синтетичний прямий прямокутник → `has_perspective = False`.
- `test_skewed_document` — прямокутник зі зміщеними кутами → `has_perspective = True`, `corners is not None`.
- `test_skew_ratio_proportional` — більший перекіс → більший `skew_ratio`.
- `test_perspective_immutable` — вхідне зображення не змінюється.

#### Клас `TestDiagnose`

- `test_returns_all_fields` — `DiagnosticResult` містить усі поля, жодне не є `None` крім `perspective_corners`.
- `test_diagnose_immutable` — вхідне зображення не змінюється.
- `test_diagnose_consistent` — два виклики на одному зображенні → однаковий результат.

#### Клас `TestRunFullAuto`

- `test_returns_correct_types` — повертає `(np.ndarray, str, dict)`.
- `test_immutable` — вхідне зображення не змінюється.
- `test_dry_run_returns_original` — `dry_run=True` → результат побайтово рівний `image.copy()`.
- `test_improves_contrast_on_narrow_histogram` — зображення з вузькою гістограмою → після Full Auto `p99-p1` більший.
- `test_improves_sharpness_on_blurry` — після `GaussianBlur` → після Full Auto `measure_sharpness` більший.
- `test_no_crash_on_plain_white` — повністю білий квадрат → не падає.
- `test_no_crash_on_plain_black` — повністю чорний квадрат → не падає.
- `test_applied_steps_logged` — якщо корекція застосована → відповідний ключ присутній в `applied_steps`.
- `test_status_message_not_empty` — `status_msg` є непорожній рядок.

### Файл `tests/unit/test_brightness_contrast.py` — додати до існуючого

**Клас `TestSmartContrastPercentile`:**
- `test_narrow_histogram_stretches` — вузька гістограма → після функції `p99-p1` більший.
- `test_wide_histogram_unchanged` — широка гістограма → результат близький до оригіналу.
- `test_strength_zero_returns_copy` — `strength=0` → рівний оригіналу.
- `test_immutable` — вхідне зображення не змінюється.

**Клас `TestContrastSCurve`:**
- `test_midtones_enhanced` — після функції дисперсія L збільшується.
- `test_shadows_not_crushed` — мінімальне L після функції вище ніж при лінійному контрасті.
- `test_highlights_not_blown` — максимальне L після функції нижче 255 якщо до цього було нижче 250.
- `test_strength_zero_minimal_effect` — `strength=0` → `gamma=1` → результат близький до оригіналу.
- `test_immutable` — вхідне зображення не змінюється.

**Клас `TestLocalContrastAdaptive`:**
- `test_detail_areas_enhanced` — в текстурованій зоні контраст зростає.
- `test_flat_areas_unchanged` — в однотонній зоні зміна мінімальна.
- `test_immutable` — вхідне зображення не змінюється.

---

## Зведена таблиця файлів і змін

| Файл | Тип | Що змінюється |
|---|---|---|
| `processing/diagnostics.py` | **Новий** | Повний діагностичний модуль |
| `processing/brightness_contrast.py` | Доповнення | 3 нові функції + 2 константи |
| `processing/pipeline.py` | Доповнення | `run_full_auto` + оновлення `run_contrast_advanced` |
| `config/app_settings.py` | Доповнення | 2 нові параметри |
| `gui/settings_window.py` | Доповнення | Нова секція Full Auto |
| `gui/main_window.py` | Модифікація | `_do_full_auto`, оновлення `_do_autofix`, текст кнопки |
| `batch/batch_processor.py` | Модифікація | Підтримка `full_auto_mode` |
| `tests/unit/test_diagnostics.py` | **Новий** | 25+ тестів |
| `tests/unit/test_brightness_contrast.py` | Доповнення | 12 нових тестів |

---

## Порядок реалізації для агента

**Крок 1** — `processing/brightness_contrast.py`: три нові функції. Незалежні, легко перевіряються.

**Крок 2** — `tests/unit/test_brightness_contrast.py`: тести для нових функцій. Переконатись що проходять.

**Крок 3** — `processing/diagnostics.py`: повний модуль без будь-яких змін в інших файлах. Функція `diagnose` повинна працювати але не застосовуватись ніде.

**Крок 4** — `tests/unit/test_diagnostics.py`: тести для діагностики. Переконатись що всі проходять.

**Крок 5** — `processing/pipeline.py`: додати `run_full_auto` з `dry_run=True` за замовчуванням спочатку. Тест `test_dry_run_returns_original` має проходити.

**Крок 6** — `config/app_settings.py` і `gui/settings_window.py`: нові параметри і UI. Перевірити що зберігається і завантажується.

**Крок 7** — `gui/main_window.py` і `batch/batch_processor.py`: інтеграція. Перевірити що старий Auto Fix працює без змін.

**Крок 8** — `processing/pipeline.py`: поступово вмикати кроки в `run_full_auto` (зняти `dry_run=True`), починаючи з різкості і яскравості (Крок 3 і 4 пайплайну), тестувати на реальних зображеннях.

**Крок 9** — Додати контраст, HDR, специфічну обробку по типу (Кроки 4-6 пайплайну).

**Крок 10** — Перспективу і shadow_remove додати останніми (Кроки 1-2 пайплайну), тільки після перевірки попередніх кроків.