# Оновлений план рефакторингу PhotoPrint

## Загальні правила для агента
- Не змінювати архітектуру модулів без потреби
- Зберігати всі існуючі публічні API якщо не сказано інше
- Після кожного завдання код має запускатись без помилок
- Коментарі у коді — українською
- Константи виносити на початок файлу

---

## Завдання H — Підняти пороги детектора тіней (5 хв)

**Файл:** `processing/shadow_remove.py`

**Що зробити:**
Змінити дві константи на початку файлу:

```python
PHOTO_COLOR_STD_THRESHOLD = 30  →  55
HSV_SATURATION_RATIO = 0.15     →  0.25
```

**Пояснення:** кольорові документи мають std каналів A/B трохи вище 30, тому детектор помилково вважає їх фото і пропускає. Підняття порогу до 55 пропускає тільки справжні фото.

Без змін у GUI і без рефакторингу — одразу дає ефект на кольорових документах.

---

## Завдання A — Виправлення `shadow_remove.py`

### A1. Виправити чорні точки що стають кольоровими

**Файл:** `processing/shadow_remove.py`

**Проблема:** коли L-канал містить абсолютно чорні пікселі (L=0), після ділення вони отримують ненульове значення L, але канали a і b залишаються незмінними. Результат — чорний піксель стає кольоровим.

**Що зробити:**
- Додати константу на початку файлу:
  ```
  L_MIN_CLAMP = 5  # мінімальне значення L перед діленням щоб уникнути артефактів
  ```
- У функції `remove_shadow()` після `l_ch, a_ch, b_ch = cv2.split(lab)` і до будь-якого ділення додати:
  ```python
  l_ch = np.maximum(l_ch, L_MIN_CLAMP)
  ```
- Те саме зробити всередині `_create_coarse_background()` — якщо `l_channel` передається туди вже після першого проходу, переконатись що мінімум теж обмежений

---

### A2. Диференційована обробка чб vs кольорових документів

**Файл:** `processing/shadow_remove.py`

**Проблема:** другий прохід (downsample) псує кольорові документи — вибілює кольорові ділянки і знищує деталі в тінях.

**Що зробити:**

Додати константи:
```
COARSE_BLEND_COLOR = 0.0       # сила другого проходу для кольорових (0 = вимкнено)
KERNEL_COLOR_MULTIPLIER = 1.5  # множник ядра першого проходу для кольорових
```

Змінити сигнатуру `remove_shadow()`:
```python
def remove_shadow(
    image: np.ndarray,
    kernel_size: int = 0,
    coarse_pass: bool = True,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
) -> np.ndarray:
```

Логіка всередині функції:
- якщо `is_color_document=True`:
  - помножити `kernel_size` на `KERNEL_COLOR_MULTIPLIER` (округлити до непарного)
  - якщо `coarse_blend == 0.0` — пропустити другий прохід повністю
  - якщо `coarse_blend > 0.0` — другий прохід застосувати з блендингом:
    ```python
    l_coarse_result = ...  # результат другого проходу
    l_norm = cv2.addWeighted(l_norm, 1.0 - coarse_blend, l_coarse_result, coarse_blend, 0)
    ```
- якщо `is_color_document=False` — поведінка як зараз (обидва проходи повністю)

Змінити сигнатуру `auto_remove_shadow()`:
```python
def auto_remove_shadow(
    image: np.ndarray,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
) -> tuple[np.ndarray, bool]:
```
Передавати ці параметри у `remove_shadow()`.

---

### A3. Послабити логіку детекції тіні

**Файл:** `processing/shadow_remove.py`

**Проблема:** функція `_detect_shadow()` має п'ять умов через AND — через це реальні документи з тінями часто не розпізнаються.

**Що зробити:**

Нова логіка `_detect_shadow()` — залишити тільки дві перевірки:
1. `laplacian_var > LAPLACIAN_BLUR_THRESHOLD` — зображення не розмите
2. `p_low < SHADOW_DETECT_THRESHOLD AND ratio < SHADOW_RATIO_THRESHOLD` — є темні ділянки з великим перепадом яскравості

**Видалити** з `_detect_shadow()`:
- перевірку `std_a > PHOTO_COLOR_STD_THRESHOLD or std_b > PHOTO_COLOR_STD_THRESHOLD`
- перевірку `saturated_ratio > HSV_SATURATION_RATIO`
- перевірку `range_l > 210`

**Пояснення:** тип документа тепер визначається зовні і передається як параметр. `_detect_shadow()` має відповідати тільки на питання "чи є нерівне освітлення", а не "чи це фото/документ".

**Видалити константи** що більше не використовуються:
```
PHOTO_COLOR_STD_THRESHOLD
HSV_SATURATION_THRESHOLD
HSV_SATURATION_RATIO
RANGE_L_THRESHOLD
```

---

### A4. Оновити виклики у `pipeline.py`

**Файл:** `processing/pipeline.py`

**Що зробити:**

У функції `run_autofix()` знайти місце де викликається `shadow_remove.auto_remove_shadow(result)` і замінити на:
```python
is_color = (doc_type == DocType.COLOR_DOCUMENT.value)
coarse_blend = settings.get("shadow_coarse_blend_color", 0.0) if settings else 0.0
result, had_shadow = shadow_remove.auto_remove_shadow(
    result,
    is_color_document=is_color,
    coarse_blend=coarse_blend,
)
```

Додати параметр `settings: dict | None = None` у сигнатуру `run_autofix()` якщо його ще немає.

Також додати перевірку режиму видалення тіней:
```python
shadow_mode = settings.get("shadow_remove_mode", "auto") if settings else "auto"
if shadow_mode == "never":
    had_shadow = False
elif shadow_mode == "always":
    result, had_shadow = shadow_remove.remove_shadow(
        result,
        is_color_document=is_color,
        coarse_blend=coarse_blend,
    )
    had_shadow = True
else:  # auto
    result, had_shadow = shadow_remove.auto_remove_shadow(
        result,
        is_color_document=is_color,
        coarse_blend=coarse_blend,
    )
```

---

## Завдання F — Тумблер для видалення тіней

### F1. Додати ComboBox у головне вікно

**Файл:** `gui/main_window.py`

У `_build_ui()` в секцію кнопок (рядок з `buttons_row`) додати після `btn_autofix`:
```python
self._combo_shadow_mode = QComboBox()
self._combo_shadow_mode.addItem("Авто", "auto")
self._combo_shadow_mode.addItem("Завжди", "always")
self._combo_shadow_mode.addItem("Ніколи", "never")
self._combo_shadow_mode.setObjectName("combo_shadow_mode")
self._combo_shadow_mode.setFixedWidth(120)
self._combo_shadow_mode.currentIndexChanged.connect(self._on_shadow_mode_changed)
```

Ініціалізувати значення з settings у `_apply_default_mode()`:
```python
shadow_mode = self._settings.get("shadow_remove_mode", "auto")
idx = self._combo_shadow_mode.findData(shadow_mode)
if idx >= 0:
    self._combo_shadow_mode.setCurrentIndex(idx)
```

Додати метод `_on_shadow_mode_changed()`:
```python
def _on_shadow_mode_changed(self, index: int):
    mode = self._combo_shadow_mode.currentData()
    self._settings["shadow_remove_mode"] = mode
    app_settings.save(self._settings)
    # Перезапускаємо Auto Fix з новим режимом
    self._do_autofix_classic()
```

Оновити `_on_settings_saved()` — додати відновлення значення комбобокса.

### F2. Додати ключ у `app_settings.py`

**Файл:** `config/app_settings.py`

Додати константу:
```python
DEFAULT_SHADOW_REMOVE_MODE = "auto"
```

У `load()` додати:
```python
"shadow_remove_mode": cfg.get("processing", "shadow_remove_mode", fallback=DEFAULT_SHADOW_REMOVE_MODE),
```

У `save()` додати:
```python
"shadow_remove_mode": settings.get("shadow_remove_mode", DEFAULT_SHADOW_REMOVE_MODE),
```

### F3. Оновити `pipeline.py`

**Файл:** `processing/pipeline.py`

У `run_autofix()` додати логіку з A4:
```python
shadow_mode = settings.get("shadow_remove_mode", "auto") if settings else "auto"
if doc_type in (DocType.BW_DOCUMENT.value, DocType.COLOR_DOCUMENT.value):
    if shadow_mode == "never":
        had_shadow = False
    elif shadow_mode == "always":
        result = shadow_remove.remove_shadow(result, is_color_document=is_color, coarse_blend=coarse_blend)
        had_shadow = True
        status_parts.append("тіні видалено (примусово)")
    else:  # auto
        result, had_shadow = shadow_remove.auto_remove_shadow(result, is_color_document=is_color, coarse_blend=coarse_blend)
        if had_shadow:
            status_parts.append("тіні видалено")
```

---

## Завдання C — Видалення Full Auto

### C1. `processing/pipeline.py`

**Видалити повністю:**
- функцію `run_full_auto()`
- функцію `_compute_adaptive_params()`

**Перевірити** що нічого більше в pipeline не залежить від цих функцій.

---

### C2. `gui/main_window.py`

**Видалити:**
- кнопку `self._btn_full_auto = QPushButton("⚡ Full Auto")` і всі посилання на неї
- метод `_do_full_auto()`
- з `buttons_row.addWidget()` — рядок з `self._btn_full_auto`
- з методу `_set_buttons_enabled()` — `self._btn_full_auto`
- з методу `_update_buttons()` — `self._btn_full_auto`

---

### C3. `config/app_settings.py`

**Видалити константи** (всі що починаються з `DEFAULT_FULL_AUTO_`):
```
DEFAULT_FULL_AUTO_MODE
DEFAULT_FULL_AUTO_MIN_GRADIENT_STRENGTH
DEFAULT_FULL_AUTO_PERSPECTIVE
DEFAULT_FULL_AUTO_HDR_ENABLED
DEFAULT_FULL_AUTO_DEFAULT_SHARPEN
DEFAULT_FULL_AUTO_SHADOW_HIGHLIGHT
DEFAULT_FULL_AUTO_CONTRAST_MODE
DEFAULT_FULL_AUTO_AUTOFIX_CONTRAST
DEFAULT_FULL_AUTO_BW_BINARY
DEFAULT_FULL_AUTO_OUTPUT_COLOR_MODE
```

**Видалити ключі** з функції `load()` — всі ключі `full_auto_*`.

**Видалити ключі** з функції `save()` — всі записи `cfg["processing"]["full_auto_*"]`.

---

### C4. `gui/settings_window.py`

**Видалити повністю** груп-бокс `fa_box = QGroupBox("Full Auto")` з усім його вмістом:
- всі `self._cb_full_auto_*`, `self._spin_full_auto_*`, `self._combo_full_auto_*` віджети
- `right.addWidget(fa_box)`

У методі `_apply_settings()` видалити всі рядки що встановлюють значення full_auto віджетів.

У методі `_collect_settings()` видалити всі ключі `full_auto_*`.

---

### C5. `batch/batch_processor.py`

У методі `run_auto()` знайти:
```python
if s.get("full_auto_mode", False):
    processed, _, _ = pipeline.run_full_auto(...)
else:
    processed, _ = pipeline.run_autofix(...)
```
Замінити на просто:
```python
processed, _ = pipeline.run_autofix(image, settings=s)
```

---

## Завдання G — Лог кроків обробки

### G1. Змінити `run_autofix()` у `pipeline.py`

**Файл:** `processing/pipeline.py`

Змінити тип повернення:
```python
def run_autofix(
    image: np.ndarray,
    ...
) -> tuple[np.ndarray, str, list[dict]]:
    """
    Повертає (результат, статус_повідомлення, список_кроків).
    
    Кожен елемент списку_кроків — dict:
        {"step": str, "applied": bool, "detail": str}
    """
```

Всередині функції замість `status_parts.append("...")` зібрати структурований список:
```python
log_entries: list[dict] = []
...
# замість status_parts.append("тіні видалено"):
log_entries.append({"step": "shadow_remove", "applied": True, "detail": "видалено"})
# замість status_parts.append("перспектива виправлена"):
log_entries.append({"step": "perspective", "applied": True, "detail": "виправлена"})
# якщо крок не застосовано:
log_entries.append({"step": "hdr", "applied": False, "detail": "вимкнено в налаштуваннях"})
```

Фінальний `status_msg` збирати з `log_entries` як і раніше:
```python
status_parts = [e["detail"] for e in log_entries if e["applied"]]
status_msg = "Auto Fix: " + ", ".join(status_parts)
return result, status_msg, log_entries
```

### G2. Оновити всі виклики `run_autofix()`

**Файли:**
- `gui/main_window.py` — `_do_autofix_classic()`:
  ```python
  result, status_msg, log_entries = pipeline.run_autofix(...)
  self._set_status(status_msg)
  self._show_log(log_entries)
  ```
- `batch/batch_processor.py` — `run_auto()`:
  ```python
  processed, status_msg, log_entries = pipeline.run_autofix(...)
  ```
  (лог не показуємо, тільки статус)
- Будь-які інші місця де викликається `run_autofix()`

### G3. Відображення логу у головному вікні

**Файл:** `gui/main_window.py`

Додати `QListWidget` або кілька `QLabel` під прев'ю.

У `_build_ui()` після `self._status` додати:
```python
self._log_widget = QListWidget()
self._log_widget.setMaximumHeight(120)
self._log_widget.setVisible(False)
center.addWidget(self._log_widget)
```

Додати метод `_show_log(log_entries: list[dict])`:
```python
def _show_log(self, log_entries: list[dict]):
    self._log_widget.clear()
    if not log_entries:
        self._log_widget.setVisible(False)
        return
    for entry in log_entries:
        icon = "✓" if entry["applied"] else "✗"
        text = f"{icon} {entry['step']}: {entry['detail']}"
        self._log_widget.addItem(text)
    self._log_widget.setVisible(True)
```

---

## Завдання E — Параметри тіней у GUI

### E1. Нові поля у `settings_window.py`

**Файл:** `gui/settings_window.py`

У існуючий груп-бокс `proc_box = QGroupBox("Auto Fix")` додати нові рядки форми після існуючих:

```python
self._cb_shadow_remove = QCheckBox()
proc_form.addRow("Видалення тіней увімкнено:", self._cb_shadow_remove)

self._spin_shadow_detect_threshold = QDoubleSpinBox()
self._spin_shadow_detect_threshold.setRange(20.0, 200.0)
self._spin_shadow_detect_threshold.setSingleStep(5.0)
self._spin_shadow_detect_threshold.setDecimals(0)
proc_form.addRow("Поріг темних ділянок p5 (0-255):", self._spin_shadow_detect_threshold)

self._spin_shadow_detect_ratio = QDoubleSpinBox()
self._spin_shadow_detect_ratio.setRange(0.05, 0.80)
self._spin_shadow_detect_ratio.setSingleStep(0.05)
self._spin_shadow_detect_ratio.setDecimals(2)
proc_form.addRow("Поріг відношення p5/p95 (0-1):", self._spin_shadow_detect_ratio)

self._spin_shadow_coarse_blend = QDoubleSpinBox()
self._spin_shadow_coarse_blend.setRange(0.0, 1.0)
self._spin_shadow_coarse_blend.setSingleStep(0.1)
self._spin_shadow_coarse_blend.setDecimals(1)
proc_form.addRow("2-й прохід для кольорових (0=вимк, 1=повний):", self._spin_shadow_coarse_blend)
```

У `_apply_settings()` додати:
```python
self._cb_shadow_remove.setChecked(s.get("shadow_remove_enabled", True))
self._spin_shadow_detect_threshold.setValue(s.get("shadow_detect_threshold", 80.0))
self._spin_shadow_detect_ratio.setValue(s.get("shadow_detect_ratio", 0.3))
self._spin_shadow_coarse_blend.setValue(s.get("shadow_coarse_blend_color", 0.0))
```

У `_collect_settings()` додати:
```python
"shadow_remove_enabled":     self._cb_shadow_remove.isChecked(),
"shadow_detect_threshold":   self._spin_shadow_detect_threshold.value(),
"shadow_detect_ratio":       self._spin_shadow_detect_ratio.value(),
"shadow_coarse_blend_color": self._spin_shadow_coarse_blend.value(),
```

---

### E2. Нові ключі у `app_settings.py`

**Файл:** `config/app_settings.py`

Додати константи:
```python
DEFAULT_SHADOW_REMOVE_ENABLED     = True
DEFAULT_SHADOW_DETECT_THRESHOLD   = 80.0
DEFAULT_SHADOW_DETECT_RATIO       = 0.3
DEFAULT_SHADOW_COARSE_BLEND_COLOR = 0.0
```

У `load()` додати:
```python
"shadow_remove_enabled":     cfg.getboolean("processing", "shadow_remove_enabled",     fallback=DEFAULT_SHADOW_REMOVE_ENABLED),
"shadow_detect_threshold":   cfg.getfloat("processing",   "shadow_detect_threshold",   fallback=DEFAULT_SHADOW_DETECT_THRESHOLD),
"shadow_detect_ratio":       cfg.getfloat("processing",   "shadow_detect_ratio",       fallback=DEFAULT_SHADOW_DETECT_RATIO),
"shadow_coarse_blend_color": cfg.getfloat("processing",   "shadow_coarse_blend_color", fallback=DEFAULT_SHADOW_COARSE_BLEND_COLOR),
```

У `save()` додати у секцію `cfg["processing"]`:
```python
"shadow_remove_enabled":     str(settings.get("shadow_remove_enabled",     DEFAULT_SHADOW_REMOVE_ENABLED)).lower(),
"shadow_detect_threshold":   str(settings.get("shadow_detect_threshold",   DEFAULT_SHADOW_DETECT_THRESHOLD)),
"shadow_detect_ratio":       str(settings.get("shadow_detect_ratio",       DEFAULT_SHADOW_DETECT_RATIO)),
"shadow_coarse_blend_color": str(settings.get("shadow_coarse_blend_color", DEFAULT_SHADOW_COARSE_BLEND_COLOR)),
```

---

### E3. Використати параметри у `shadow_remove.py`

**Файл:** `processing/shadow_remove.py`

У `_detect_shadow()` замінити захардкоджені значення на параметри:

Змінити сигнатуру:
```python
def _detect_shadow(
    image: np.ndarray,
    threshold: float = SHADOW_DETECT_THRESHOLD,
    ratio: float = SHADOW_RATIO_THRESHOLD,
) -> bool:
```

Використовувати `threshold` і `ratio` замість констант всередині функції.

У `auto_remove_shadow()` прийняти і передати ці параметри:
```python
def auto_remove_shadow(
    image: np.ndarray,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
    detect_threshold: float = SHADOW_DETECT_THRESHOLD,
    detect_ratio: float = SHADOW_RATIO_THRESHOLD,
) -> tuple[np.ndarray, bool]:
```

У `pipeline.py` у `run_autofix()` передавати значення з settings:
```python
result, had_shadow = shadow_remove.auto_remove_shadow(
    result,
    is_color_document=is_color,
    coarse_blend=settings.get("shadow_coarse_blend_color", 0.0),
    detect_threshold=settings.get("shadow_detect_threshold", 80.0),
    detect_ratio=settings.get("shadow_detect_ratio", 0.3),
)
```

---

## Завдання B — Авто-перспектива з deskew

### B1. Новий модуль `processing/deskew.py`

**Файл:** `processing/deskew.py` (створити новий)

**Константи:**
```python
DESKEW_MIN_ANGLE = 0.5          # менше цього — не повертати
DESKEW_MAX_ANGLE = 45.0         # більше цього — ігнорувати як шум
DESKEW_HOUGH_THRESHOLD = 50     # мінімальна кількість голосів для лінії
DESKEW_HOUGH_MIN_LENGTH = 100   # мінімальна довжина лінії в пікселях
DESKEW_HOUGH_MAX_GAP = 10       # максимальний розрив у лінії
DESKEW_RESIZE_MAX = 800         # розмір для аналізу (швидкість)
DESKEW_ANGLE_FILTER_LOW = -45.0 # нижня межа кута для фільтрації
DESKEW_ANGLE_FILTER_HIGH = 45.0 # верхня межа кута
```

**Функція `measure_skew_angle(image: np.ndarray) -> float`:**
1. Зменшити зображення до `DESKEW_RESIZE_MAX` по більшій стороні
2. Конвертувати в grayscale
3. Застосувати adaptive threshold (`cv2.adaptiveThreshold`, `THRESH_BINARY_INV`) щоб виділити темний текст на світлому фоні
4. Застосувати `cv2.HoughLinesP` з параметрами з констант
5. Для кожної знайденої лінії обчислити кут через `np.arctan2(dy, dx)` і конвертувати в градуси
6. Відфільтрувати кути: залишити тільки ті що в діапазоні `-DESKEW_ANGLE_FILTER_HIGH .. DESKEW_ANGLE_FILTER_HIGH`
7. Якщо ліній менше 3 — повернути `0.0`
8. Повернути медіану кутів (`np.median`)

**Функція `apply_deskew(image: np.ndarray, angle: float) -> np.ndarray`:**
- якщо `abs(angle) < DESKEW_MIN_ANGLE` — повернути `image.copy()`
- обчислити центр зображення
- побудувати матрицю повороту: `cv2.getRotationMatrix2D(center, -angle, 1.0)`
- обчислити новий розмір canvas щоб зображення не обрізалось після повороту (через тригонометрію)
- застосувати `cv2.warpAffine` з `borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)`
- повернути результат

---

### B2. Нова функція `run_perspective_auto_smart` у `pipeline.py`

**Файл:** `processing/pipeline.py`

**Додати імпорт:**
```python
from processing import deskew as deskew_module
```

**Нова функція:**
```python
def run_perspective_auto_smart(
    image: np.ndarray,
    settings: dict | None = None,
) -> tuple[np.ndarray, str]:
```

**Логіка функції:**
1. Виміряти кут: `angle = deskew_module.measure_skew_angle(image)`
2. Спробувати знайти кути: `corners = perspective.auto_detect_corners(image)`
3. Якщо `corners` знайдено — перевірити через `perspective.detect_skewed_sides()` чи є реальне викривлення перспективи
4. Прийняти рішення:

   | corners | skewed_sides | angle | Дія |
   |---------|-------------|-------|-----|
   | None | — | < 0.5° | нічого, статус "перспектива не потрібна" |
   | None | — | ≥ 0.5° | тільки `apply_deskew(image, angle)` |
   | знайдено | жодна | < 0.5° | нічого |
   | знайдено | жодна | ≥ 0.5° | тільки deskew |
   | знайдено | є хоча б одна | будь-який | warp через `perspective.apply_correction()`, потім deskew результату |

5. Повернути `(result, status_message)`

---

### B3. Підключити у `pipeline.py` та `main_window.py`

**Файл:** `processing/pipeline.py`

У `run_autofix()` знайти блок:
```python
if use_perspective:
    corrected, found = perspective.auto_correct(result) ...
```
Замінити на виклик `run_perspective_auto_smart(result, settings)`.

**Файл:** `gui/main_window.py`

У методі `_do_persp_auto()` знайти виклик `pipeline.detect_corners()` / `pipeline.run_perspective_auto()` і замінити на:
```python
result, status = pipeline.run_perspective_auto_smart(self._base, self._settings)
```
Оновити логіку відображення результату відповідно.

---

## Завдання D — Пресети стратегій замість вільного переміщення

### D1. Що прибрати з попереднього плану

З оригінального завдання D **прибрати повністю**:
- кнопки ▲ ▼ для переміщення кроків (`_move_up()`, `_move_down()`)
- клас `PipelineStepsWidget` з `QListWidget` і переміщенням
- логіку циклу по довільному порядку у `pipeline.py`
- збереження `pipeline_steps_order` у settings
- константи `DEFAULT_PIPELINE_STEPS_ORDER` і `DEFAULT_PIPELINE_STEPS_ENABLED`

### D2. Пресети стратегій

**Файл:** `gui/settings_window.py`

Створити новий клас `StrategyPresetWidget(QWidget)`:

```python
# Константи для пресетів
PRESETS = {
    "doc_bw": {
        "label": "Документ (чб)",
        "steps": ["shadow_remove", "perspective", "brightness", "contrast", "sharpen", "grayscale", "white_background"],
    },
    "doc_color": {
        "label": "Документ (кольоровий)",
        "steps": ["shadow_remove", "perspective", "brightness", "contrast", "sharpen", "white_background"],
    },
    "photo": {
        "label": "Фото",
        "steps": ["perspective", "hdr", "sharpen"],
    },
    "geometry": {
        "label": "Тільки геометрія",
        "steps": ["perspective"],
    },
    "custom": {
        "label": "Власний",
        "steps": None,  # користувацькі чекбокси
    },
}
```

**Віджет містить:**
- `QComboBox` з п'ятьма пресетами
- `QListWidget` з `QCheckBox` для кожного кроку (видимий тільки в режимі "Власний")

**Публічні методи:**
- `get_enabled_steps() -> list[str]` — повертає список ключів увімкнених кроків у фіксованому порядку
- `set_state(preset: str, enabled: list[str])` — встановлює стан

**Фіксований порядок кроків (завжди однаковий):**
```python
PIPELINE_STEPS_FIXED_ORDER = [
    ("shadow_remove",    "Видалення тіней"),
    ("perspective",      "Авто-перспектива"),
    ("brightness",       "Авто-яскравість"),
    ("contrast",         "Авто-контраст"),
    ("hdr",              "HDR"),
    ("sharpen",          "Різкість"),
    ("grayscale",        "Grayscale / бінаризація"),
    ("white_background", "Білий фон"),
]
```

### D3. Підключення у `SettingsWindow`

**Файл:** `gui/settings_window.py`

У `_build_ui()` додати новий групбокс:
```python
preset_box = QGroupBox("Стратегія обробки")
preset_box.setStyleSheet(GROUPBOX_STYLE)
preset_layout = QVBoxLayout(preset_box)
self._preset_widget = StrategyPresetWidget()
preset_layout.addWidget(self._preset_widget)
left.addWidget(preset_box)
```

У `_apply_settings()` додати:
```python
preset = s.get("pipeline_preset", "doc_bw")
enabled_str = s.get("pipeline_steps_enabled", "")
enabled = [k.strip() for k in enabled_str.split(",") if k.strip()] if enabled_str else None
self._preset_widget.set_state(preset, enabled)
```

У `_collect_settings()` додати:
```python
"pipeline_preset": self._preset_widget.get_preset(),
"pipeline_steps_enabled": ",".join(self._preset_widget.get_enabled_steps()),
```

### D4. Зберігання у `app_settings.py`

**Файл:** `config/app_settings.py`

Додати константи:
```python
DEFAULT_PIPELINE_PRESET = "doc_bw"
DEFAULT_PIPELINE_STEPS_ENABLED = "shadow_remove,perspective,brightness,contrast,hdr,sharpen,grayscale,white_background"
```

У `load()` додати:
```python
"pipeline_preset":       cfg.get("processing", "pipeline_preset",       fallback=DEFAULT_PIPELINE_PRESET),
"pipeline_steps_enabled": cfg.get("processing", "pipeline_steps_enabled", fallback=DEFAULT_PIPELINE_STEPS_ENABLED),
```

У `save()` додати:
```python
cfg["processing"]["pipeline_preset"]        = settings.get("pipeline_preset",       DEFAULT_PIPELINE_PRESET)
cfg["processing"]["pipeline_steps_enabled"] = settings.get("pipeline_steps_enabled", DEFAULT_PIPELINE_STEPS_ENABLED)
```

### D5. Виконання кроків у `pipeline.py`

**Файл:** `processing/pipeline.py`

На початку `run_autofix()`:
```python
preset = settings.get("pipeline_preset", "doc_bw") if settings else "doc_bw"
steps_enabled_str = settings.get("pipeline_steps_enabled", None) if settings else None
```

Якщо preset != "custom" — використовувати фіксований список кроків для пресета.
Якщо preset == "custom" — парсити `steps_enabled_str`.

Цикл по фіксованому порядку:
```python
for step_key, _ in PIPELINE_STEPS_FIXED_ORDER:
    if step_key not in steps_enabled:
        continue
    if step_key == "shadow_remove":
        ...
    elif step_key == "perspective":
        ...
    # і так далі
```

**Важливо:** класифікація типу документа (`doc_classifier.classify()`) виконується **до** циклу, завжди, незалежно від налаштувань.

---

## Порядок виконання завдань

```
H (підняти пороги — 5 хвилин, миттєвий ефект)
  ↓
A (shadow_remove виправлення — чорні точки, диф. обробка, детектор)
  ↓
F (тумблер авто/завжди/ніколи у GUI)
  ↓
C (видалення Full Auto)
  ↓
G (лог кроків обробки)
  ↓
E (параметри тіней у GUI)
  ↓
B (deskew модуль + smart perspective)
  ↓
D (пресети стратегій замість вільного переміщення)
```

Кожне завдання незалежне від наступного і програма має працювати після кожного кроку.