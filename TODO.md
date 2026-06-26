# TASK.md — План для агента кодування: PhotoPrint

> **Правила:**
> - Після кожного кроку агент ставить `[x]` у цьому файлі.
> - Не писати код поза межами зазначених файлів.
> - Не рефакторити те, чого немає в задачі.
> - Слайдер різкості **не чіпати** (виключено з плану).

---

## Задача 1 — GUI: панель «ДО» ніколи не змінюється після першого завантаження файлу

**Пріоритет:** КРИТИЧНИЙ  
**Файл:** `gui/main_window.py`

### Контекст

Зараз `self._preview.set_before(...)` викликається у кількох місцях після першого завантаження, що змінює оригінальну панель. Це суперечить вимозі. Єдині дозволені місця виклику `set_before` — `_on_queue_selection` і `_load_next_manual`, і тільки з `self._orig`.

### Кроки

- [ ] Знайти всі виклики `self._preview.set_before(...)` у файлі та скласти список.
- [ ] У методі `_do_persp_auto` → у внутрішньому callback `_on_done` — **видалити** рядок:
  ```
  self._preview.set_before(image_utils.make_preview(result))
  ```
  Панель «ДО» не оновлюється після авто-перспективи. Змінюється лише «ПІСЛЯ».
- [ ] У методі `_do_persp_reset` — **видалити** рядок:
  ```
  self._preview.set_before(image_utils.make_preview(self._base))
  ```
  Після скидання перспективи «ДО» не оновлюється.
- [ ] У методі `_do_reset_all` — рядок:
  ```
  self._preview.set_before(image_utils.make_preview(self._orig))
  ```
  Залишити, бо він використовує `self._orig` — це коректно (скидання до оригіналу).
- [ ] Перевірити метод `_on_shadow_mode_changed`: він викликає `_do_autofix_classic()` → перевірити що `_do_autofix_classic` у своєму `_on_done` НЕ викликає `set_before`. Судячи з коду — не викликає, але підтвердити.
- [ ] Перевірити метод `_show_perspective_points`: він викликає `self._preview.enable_perspective_edit(pts)` на панелі `_before` (на ній малюються точки). Це технічно не `set_before`, але візуально змінює панель «ДО» — **залишити** (це інтерактивний режим редагування, де точки мають бути саме на оригіналі).
- [ ] Переконатися: `set_before` залишається тільки у двох місцях:
  - `_on_queue_selection` — `self._preview.set_before(prev)` де `prev` будується з `self._orig`
  - `_load_next_manual` — `self._preview.set_before(prev)` де `prev` будується з щойно завантаженого `img`
- [ ] Відмітити виконання в TASK.md.

---

## Задача 2 — HDR: ручний слайдер має давати видимий ефект

**Пріоритет:** ВИСОКИЙ  
**Файли:** `processing/hdr.py`, `processing/pipeline.py`

### Контекст

Три рівні пригнічення складаються і дають ефективну силу ~0.05 від заданого:
1. `_auto_strength_factor` — якщо діапазон L вузький (білий документ) → множник 0.25
2. `detail_map.detail_mask` — на рівному фоні маска ~0 → ефект CLAHE не застосовується
3. `_apply_coring` — дрібні різниці < 1.5L обнуляються

В авто-режимі ці обмеження **доречні**. У ручному режимі — **заважають**.

### Кроки

**`processing/hdr.py`:**

- [ ] Додати константу:
  ```
  HDR_MANUAL_NOISE_FLOOR = 0.3   # знижений coring для ручного режиму
  ```
- [ ] У сигнатуру функції `apply_adaptive(...)` додати параметр `manual_mode: bool = False`.
- [ ] Усередині `apply_adaptive`:
  - Якщо `manual_mode=True` — **не викликати** `_auto_strength_factor`, тобто `effective_strength = strength` (без множника).
  - Якщо `manual_mode=True` — передавати `noise_floor=HDR_MANUAL_NOISE_FLOOR` у `_apply_coring` замість `HDR_NOISE_FLOOR`.
  - Якщо `manual_mode=True` — **не застосовувати** `detail_mask` (пропустити блок `if auto_detail`), щоб ефект діяв по всьому зображенню.
- [ ] У сигнатуру функції `apply(...)` додати параметр `manual_mode: bool = False`.
- [ ] У тілі `apply()` — передати `manual_mode` у виклик `apply_adaptive(...)`.

**`processing/pipeline.py`:**

- [ ] У функції `run_manual_adjustments(...)` — у виклику `hdr.apply(result, strength=hdr_strength)` додати `manual_mode=True`:
  ```python
  result = hdr.apply(result, strength=hdr_strength, manual_mode=True)
  ```
- [ ] У функції `run_hdr(...)` (публічна обгортка) — перевірити чи вона використовується у ручному режимі GUI; якщо так — також додати `manual_mode=True`.
- [ ] У `run_autofix` та авто-pipeline — **не змінювати** (залишити без `manual_mode`, тобто `False` за замовчуванням).
- [ ] Відмітити виконання в TASK.md.

---

## Задача 3 — Shadow Remove: `background_uniformity` як основний критерій

**Пріоритет:** ВИСОКИЙ  
**Файли:** `processing/pipeline.py`

### Контекст

Поточна логіка: `doc_type in (bw_document, color_document)` → запускати shadow_remove. Але тип документа не є достатнім предиктором. Головний критерій — чи є фон однорідним. Це вже є в `diagnostics.measure_background_metrics()`.

### Кроки

- [ ] На початку функції `run_autofix(...)` — до будь-якої обробки зображення — додати виклик:
  ```python
  from processing import diagnostics as _diag
  _bg_uniformity, _detail_density = _diag.measure_background_metrics(image)
  ```
  Зберегти результат у локальні змінні `_bg_uniformity` і `_detail_density`.

- [ ] Знайти блок `elif step_key == "shadow_remove":` у циклі по кроках.

- [ ] Замінити внутрішню логіку прийняття рішення «запускати чи ні» за схемою:

  **Якщо `shadow_mode == "never"`** → не запускати (поточна логіка, без змін).

  **Якщо `shadow_mode == "always"`** → запускати завжди (поточна логіка, без змін).

  **Якщо `shadow_mode == "auto"`** → нова логіка:
  - Якщо `_bg_uniformity > 0.55` → **запускати** shadow_remove незалежно від `doc_type` (навіть для фото з однорідним фоном).
  - Якщо `_bg_uniformity < 0.30` → **не запускати** ні за яких умов (складний фон, ризик артефактів).
  - Якщо `0.30 ≤ _bg_uniformity ≤ 0.55` → залишити поточну логіку: запускати тільки для `bw_document` і `color_document`.

- [ ] Параметр `is_color_document` для `shadow_remove.auto_remove_shadow` — залишити на основі `doc_type` (як зараз).

- [ ] У `log_entries` для кроку `shadow_remove` — додати до поля `"detail"` значення `_bg_uniformity` для діагностики: наприклад `f"тіні видалено (uniformity={_bg_uniformity:.2f})"`.

- [ ] Відмітити виконання в TASK.md.

---

## Задача 4 — Класифікатор: визначення умов зйомки

**Пріоритет:** СЕРЕДНІЙ  
**Файл:** `processing/doc_classifier.py`

### Контекст

Потрібна нова функція, що визначає **умови зйомки** незалежно від типу документа. Значення: `"screen_capture"`, `"phone_camera"`, `"flat_uniform"`, `"angled"`, `"unknown"`.

### Кроки

- [ ] Додати константи у верхній частині файлу:
  ```python
  # Умови зйомки
  SCREEN_EDGE_DARK_THRESHOLD = 40    # пікселі темніші за це — підозра на рамку UI
  SCREEN_EDGE_STRIP_WIDTH = 15       # ширина смуги по краях для аналізу
  SCREEN_EDGE_DARK_RATIO = 0.6       # якщо >60% пікселів смуги темні → screen_capture
  PHONE_WARM_B_THRESHOLD = 133       # b-канал LAB > цього → теплий відтінок фону
  PHONE_COOL_B_THRESHOLD = 123       # b-канал LAB < цього → холодний відтінок фону
  FLAT_UNIFORM_BG_THRESHOLD = 0.60   # background_uniformity > цього → flat_uniform
  ```

- [ ] Додати допоміжну функцію `_detect_screen_capture(small: np.ndarray) -> bool`:
  - Взяти верхню, нижню, ліву, праву смуги шириною `SCREEN_EDGE_STRIP_WIDTH` пікселів від країв `small`.
  - Конвертувати в grayscale.
  - Для кожної смуги порахувати частку пікселів < `SCREEN_EDGE_DARK_THRESHOLD`.
  - Якщо хоча б для **двох** смуг ця частка > `SCREEN_EDGE_DARK_RATIO` → повернути `True`.

- [ ] Додати допоміжну функцію `_detect_phone_camera(small: np.ndarray, bg_uniformity: float) -> bool`:
  - Якщо `bg_uniformity < 0.4` → повернути `False` (фон нерівномірний, аналіз кольору ненадійний).
  - Конвертувати в LAB, взяти b-канал.
  - Знайти пікселі де L > 180 (світлий фон) — маска фону.
  - Якщо маска має < 5% пікселів → повернути `False`.
  - Порахувати медіану b-каналу на пікселях фону.
  - Якщо медіана > `PHONE_WARM_B_THRESHOLD` або < `PHONE_COOL_B_THRESHOLD` → повернути `True`.

- [ ] Додати публічну функцію:
  ```python
  def classify_capture_conditions(
      image: np.ndarray,
      background_uniformity: float = 0.5,
  ) -> str:
  ```
  Логіка (в порядку перевірки):
  1. `small = cv2.resize(image, ...)` до 300px по більшій стороні (для швидкості).
  2. Якщо `_detect_screen_capture(small)` → повернути `"screen_capture"`.
  3. Якщо `background_uniformity > FLAT_UNIFORM_BG_THRESHOLD` → перевірити `_detect_phone_camera(small, background_uniformity)`: якщо True → `"phone_camera"`, інакше → `"flat_uniform"`.
  4. Інакше → `"unknown"`.

- [ ] Додати `CaptureConditions` як рядкові константи у верхній частині файлу (для уникнення опечаток у pipeline):
  ```python
  CAPTURE_SCREEN = "screen_capture"
  CAPTURE_PHONE = "phone_camera"
  CAPTURE_FLAT = "flat_uniform"
  CAPTURE_UNKNOWN = "unknown"
  ```

- [ ] Відмітити виконання в TASK.md.

---

## Задача 5 — Pipeline: використовувати `capture_conditions` у рішеннях

**Пріоритет:** СЕРЕДНІЙ  
**Файл:** `processing/pipeline.py`

### Контекст

Після появи `classify_capture_conditions` — підключити її до `run_autofix` для прийняття розумніших рішень щодо shadow_remove та інших кроків.

### Кроки

- [ ] На початку `run_autofix` — після виклику `measure_background_metrics` (Задача 3) — додати:
  ```python
  from processing.doc_classifier import classify_capture_conditions
  _capture_cond = classify_capture_conditions(image, background_uniformity=_bg_uniformity)
  ```

- [ ] У блоці `elif step_key == "shadow_remove"` — доповнити логіку `"auto"` правилом для `screen_capture`:
  - Якщо `_capture_cond == "screen_capture"` → **не запускати** shadow_remove навіть якщо `_bg_uniformity > 0.55` (екран уже рівномірний, shadow_remove зіпсує зображення).
  - Це має бути першою перевіркою всередині `"auto"` гілки — до перевірки `_bg_uniformity`.

- [ ] Для `_capture_cond == "phone_camera"` і `shadow_mode == "auto"`:
  - **Якщо** shadow_remove запускається (тобто `_bg_uniformity > 0.30`) — після виконання shadow_remove додати нейтралізацію кольорового відтінку:
    - Конвертувати результат у LAB.
    - На пікселях де L > 180 (світлий фон) зсунути b-канал і a-канал ближче до 128 (нейтральне).
    - Ступінь зсуву: 30% від відхилення (м'яка корекція, не повна нейтралізація).
    - Реалізувати безпосередньо у pipeline як inline-код (не виносити в окремий модуль поки).

- [ ] У блоці `elif step_key == "perspective"` — якщо `_capture_cond == "screen_capture"`:
  - **Форсувати** запуск perspective навіть якщо `use_perspective=False` у налаштуваннях.
  - Додати в log_entries: `{"step": "perspective_forced", "applied": True, "detail": "screen_capture detected"}`.

- [ ] У кожному `log_entries.append(...)` де є `"detail"` — додати `_capture_cond` до рядка деталей де це доречно:
  Наприклад у записі `doc_type`: `f"ч-б документ ({_capture_cond})"`.

- [ ] Відмітити виконання в TASK.md.

---

## Задача 6 — GUI: розгорнутий лог обробки у статусному рядку

**Пріоритет:** НИЗЬКИЙ  
**Файли:** `gui/main_window.py`

### Контекст

`run_autofix` вже повертає `log_entries: list[dict]`. Потрібно відображати ці дані користувачу у зрозумілому вигляді.

### Кроки

- [ ] У методі `_do_autofix_classic` → у внутрішньому `_on_done(payload)` — розпакувати `log_entries`:
  ```python
  result, status_msg, log_entries = payload
  ```
  (Вже так і є — перевірити.)

- [ ] Побудувати короткий рядок з `log_entries` для статусного рядка:
  - Відфільтрувати тільки `entry["applied"] == True`.
  - Відфільтрувати кроки `doc_type`, `color_mode` — вивести окремо на початку.
  - Решта кроків — через ` | `.
  - Приклад: `«Розпізнано: ч-б документ (phone_camera) | Тіні: видалено (uniformity=0.71) | Контраст: +0.15 | Різкість: 0.4»`

- [ ] Передати цей рядок у `self._set_status(...)` замість поточного `status_msg` (або додатково до нього).

- [ ] У `_set_status` — зараз метод завжди викликає `self.statusBar().showMessage(text, timeout_ms)`. Замість цього:
  - Якщо `text` починається з `"Auto Fix:"` — встановити колір тексту статус-бару зелений (`color: #006600`).
  - Якщо текст містить `"Помилка"` — колір червоний.
  - Інакше — стандартний колір.
  - Реалізувати через `self.statusBar().setStyleSheet(...)` перед `showMessage`.

- [ ] Після закінчення авто-обробки повернути стандартний стиль статус-бару (скинути кольоровий override).

- [ ] Відмітити виконання в TASK.md.

---

## Порядок виконання

| № | Задача | Пріоритет | Статус |
|---|--------|-----------|--------|
| 1 | GUI: панель «ДО» не змінюється | КРИТИЧНИЙ | - [ ] |
| 2 | HDR: ручний режим | ВИСОКИЙ | - [ ] |
| 3 | Shadow Remove: background_uniformity | ВИСОКИЙ | - [ ] |
| 4 | Класифікатор: capture_conditions | СЕРЕДНІЙ | - [ ] |
| 5 | Pipeline: використовувати capture_conditions | СЕРЕДНІЙ | - [ ] |
| 6 | GUI: лог обробки | НИЗЬКИЙ | - [ ] |

---

## Залежності між задачами

```
Задача 1  →  незалежна, виконувати першою
Задача 2  →  незалежна від решти
Задача 3  →  незалежна від Задачі 4 і 5, але дає основу для них
Задача 4  →  незалежна (новий код у класифікаторі)
Задача 5  →  залежить від Задачі 3 та Задачі 4 (мають бути виконані раніше)
Задача 6  →  залежить від Задачі 3 (log_entries збагачуються там)
```

---

## Обмеження для агента

- Слайдер **різкості** (`sharpen.py`) — **не чіпати**.
- Файл `processing/autofix.py` — не змінювати у рамках цих задач.
- Файл `processing/shadow_remove.py` — не змінювати алгоритм, тільки логіку виклику в pipeline.
- Юніт-тести — не писати (виходить за рамки задач).
- При будь-якій неоднозначності в коді — зупинитись і повідомити, не вгадувати.