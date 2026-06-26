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

### Кроки

- [x] Знайти всі виклики `self._preview.set_before(...)` у файлі та скласти список.
- [x] Видалити `set_before` у `_do_persp_auto` → `_on_done`
- [x] Видалити `set_before` у `_do_persp_reset`
- [x] `_do_reset_all` — залишити (коректно, використовує `self._orig`)
- [x] Перевірити `_on_shadow_mode_changed` — не викликає `set_before`
- [x] Перевірити `_show_perspective_points` — залишити (інтерактивне редагування)
- [x] Переконатися: `set_before` залишається тільки у `_on_queue_selection` і `_load_next_manual`

---

## Задача 2 — HDR: ручний слайдер має давати видимий ефект

**Пріоритет:** ВИСОКИЙ  
**Файли:** `processing/hdr.py`, `processing/pipeline.py`

### Кроки

- [x] Додати константу `HDR_MANUAL_NOISE_FLOOR = 0.3` у `hdr.py`
- [x] Додати параметр `manual_mode: bool = False` у `apply_adaptive`
- [x] У `apply_adaptive`: при `manual_mode=True` — не викликати `_auto_strength_factor`, пропустити `detail_mask`, використати `HDR_MANUAL_NOISE_FLOOR`
- [x] Додати параметр `manual_mode: bool = False` у `apply` і передати в `apply_adaptive`
- [x] У `pipeline.py` `run_manual_adjustments`: додати `manual_mode=True` до виклику `hdr.apply`

---

## Задача 3 — Shadow Remove: `background_uniformity` як основний критерій

**Пріоритет:** ВИСОКИЙ  
**Файли:** `processing/pipeline.py`

### Кроки

- [x] На початку `run_autofix` додати виклик `measure_background_metrics(image)`
- [x] У блоці `elif step_key == "shadow_remove":` змінити логіку для `"auto"`:
  - `_bg_uniformity > 0.55` → запускати завжди
  - `_bg_uniformity < 0.30` → не запускати
  - `0.30 ≤ _bg_uniformity ≤ 0.55` → поточна логіка (за типом документа)
- [x] Додати `_bg_uniformity` у `log_entries`

---

## Задача 4 — Класифікатор: визначення умов зйомки

**Пріоритет:** СЕРЕДНІЙ  
**Файл:** `processing/doc_classifier.py`

### Кроки

- [x] Додати константи `SCREEN_EDGE_DARK_THRESHOLD`, `SCREEN_EDGE_STRIP_WIDTH`, `SCREEN_EDGE_DARK_RATIO`, `PHONE_WARM_B_THRESHOLD`, `PHONE_COOL_B_THRESHOLD`, `FLAT_UNIFORM_BG_THRESHOLD`
- [x] Додати константи `CAPTURE_SCREEN`, `CAPTURE_PHONE`, `CAPTURE_FLAT`, `CAPTURE_UNKNOWN`
- [x] Додати функцію `_detect_screen_capture(small) -> bool`
- [x] Додати функцію `_detect_phone_camera(small, bg_uniformity) -> bool`
- [x] Додати публічну функцію `classify_capture_conditions(image, background_uniformity) -> str`

---

## Задача 5 — Pipeline: використовувати `capture_conditions` у рішеннях

**Пріоритет:** СЕРЕДНІЙ  
**Файл:** `processing/pipeline.py`

### Кроки

- [x] Після `measure_background_metrics` викликати `classify_capture_conditions`
- [x] У `shadow_remove` при `screen_capture` → не запускати
- [x] У `shadow_remove` при `phone_camera` → після видалення тіней додати нейтралізацію кольорового відтінку
- [x] У `perspective` при `screen_capture` → форсувати запуск
- [x] Додати `_capture_cond` у `doc_type` log_entries

---

## Задача 6 — GUI: розгорнутий лог обробки у статусному рядку

**Пріоритет:** НИЗЬКИЙ  
**Файли:** `gui/main_window.py`

### Кроки

- [x] Побудувати рядок з `log_entries`: `doc_type`/`color_mode` на початку, решта через ` | `
- [x] Передати рядок у `_set_status`
- [x] У `_set_status`: зелений колір для `"Auto Fix:"`, червоний для `"Помилка"`, стандартний інакше
- [x] Додати `_reset_status_style()` — скидання стилю через 1.5с після авто-обробки

---

## Порядок виконання

| № | Задача | Пріоритет | Статус |
|---|--------|-----------|--------|
| 1 | GUI: панель «ДО» не змінюється | КРИТИЧНИЙ | [x] |
| 2 | HDR: ручний режим | ВИСОКИЙ | [x] |
| 3 | Shadow Remove: background_uniformity | ВИСОКИЙ | [x] |
| 4 | Класифікатор: capture_conditions | СЕРЕДНІЙ | [x] |
| 5 | Pipeline: використовувати capture_conditions | СЕРЕДНІЙ | [x] |
| 6 | GUI: лог обробки | НИЗЬКИЙ | [x] |

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