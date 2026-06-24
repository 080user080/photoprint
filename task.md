# TASK: Виправлення видалення тіней

## Крок 1 — Виправити pipeline.py: "Завжди" не залежить від типу документа

**Файл:** `processing/pipeline.py`

**Проблема:** Блок `step_key == "shadow_remove"` повністю загорнутий в:
```python
if doc_type in (DocType.BW_DOCUMENT.value, DocType.COLOR_DOCUMENT.value):
```
Через це режим "Завжди" ігнорується для `photo` та `flat_background`.

**Що зробити:**

Переписати блок `step_key == "shadow_remove"` так, щоб режим `"always"`
виконувався для БУДЬ-ЯКОГО типу документа:

```python
elif step_key == "shadow_remove":
    shadow_mode = settings.get("shadow_remove_mode", "auto") if settings else "auto"
    coarse_blend = settings.get("shadow_coarse_blend_color", 0.0) if settings else 0.0
    detect_threshold = settings.get("shadow_detect_threshold", 80.0) if settings else 80.0
    detect_ratio = settings.get("shadow_detect_ratio", 0.3) if settings else 0.3
    is_color = (doc_type == DocType.COLOR_DOCUMENT.value)
    use_bgr_mode = settings.get("shadow_bgr_mode", False) if settings else False

    if shadow_mode == "always":
        # Примусово — для будь-якого типу документа
        result = shadow_remove.remove_shadow(
            result,
            is_color_document=is_color,
            coarse_blend=coarse_blend,
            bgr_mode=use_bgr_mode,
        )
        log_entries.append({"step": "shadow_remove", "applied": True,
                            "detail": "тіні видалено (примусово)"})
    elif shadow_mode == "never":
        pass  # нічого не робимо
    else:  # auto — тільки для документів, не для фото
        if doc_type in (DocType.BW_DOCUMENT.value, DocType.COLOR_DOCUMENT.value):
            result, had_shadow = shadow_remove.auto_remove_shadow(
                result,
                is_color_document=is_color,
                coarse_blend=coarse_blend,
                detect_threshold=detect_threshold,
                detect_ratio=detect_ratio,
                bgr_mode=use_bgr_mode,
            )
            if had_shadow:
                log_entries.append({"step": "shadow_remove", "applied": True,
                                   "detail": "тіні видалено"})
    # Висвітлення тіней — додаткове підсвічування (залишається як є)
    if shadow_highlight_strength > EPSILON:
        result = shadow_highlight.apply_shadow_highlight(
            result, strength=shadow_highlight_strength
        )
        log_entries.append({"step": "shadow_highlight", "applied": True,
                           "detail": f"підсвічування {shadow_highlight_strength:.2f}"})
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 2 — Додати BGR-алгоритм у processing/shadow_remove.py

**Файл:** `processing/shadow_remove.py`

**Що зробити:**

Додати константу на початку файлу:
```python
BGR_MODE_DEFAULT = False   # якщо True — обробляє кожен канал BGR окремо
```

Додати нову внутрішню функцію `_remove_shadow_bgr()` на основі алгоритму
з `shadow_remove_gui.py` (обробка трьох каналів окремо):

```python
def _remove_shadow_bgr(
    image: np.ndarray,
    kernel_size: int = 0,
    coarse_pass: bool = True,
) -> np.ndarray:
    """
    Видаляє тіні через морфологічну обробку кожного BGR-каналу окремо.
    Алгоритм з shadow_remove_gui.py — дає кращі результати для деяких
    типів документів з рівномірним кольоровим фоном.
    """
    if kernel_size == 0:
        kernel_size = _auto_kernel_size(image)
    kernel_size = max(kernel_size | 1, MORPH_KERNEL_MIN)

    channels = cv2.split(image)
    result_channels = []

    for ch in channels:
        ch = np.maximum(ch, L_MIN_CLAMP)
        bg = _create_background_model(ch, kernel_size)
        bg_f = bg.astype(np.float32) + DIVIDE_EPSILON
        normed = cv2.divide(ch.astype(np.float32), bg_f, scale=DIVIDE_SCALE)
        normed = np.clip(normed, 0.0, 255.0).astype(np.uint8)

        if coarse_pass and COARSE_PASS_ENABLED:
            coarse_bg = _create_coarse_background(normed)
            coarse_bg_f = coarse_bg.astype(np.float32) + COARSE_DIVIDE_EPSILON
            normed2 = cv2.divide(normed.astype(np.float32), coarse_bg_f, scale=DIVIDE_SCALE)
            normed = np.clip(normed2, 0.0, 255.0).astype(np.uint8)

        result_channels.append(normed)

    return cv2.merge(result_channels)
```

Оновити сигнатуру `remove_shadow()` — додати параметр `bgr_mode`:
```python
def remove_shadow(
    image: np.ndarray,
    kernel_size: int = 0,
    coarse_pass: bool = True,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
    bgr_mode: bool = BGR_MODE_DEFAULT,
) -> np.ndarray:
```

На початку функції `remove_shadow()` додати розгалуження:
```python
if bgr_mode:
    return _remove_shadow_bgr(image, kernel_size=kernel_size, coarse_pass=coarse_pass)
```

Оновити сигнатуру `auto_remove_shadow()` — додати `bgr_mode` і передавати в `remove_shadow()`:
```python
def auto_remove_shadow(
    image: np.ndarray,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
    detect_threshold: float = SHADOW_DETECT_THRESHOLD,
    detect_ratio: float = SHADOW_RATIO_THRESHOLD,
    bgr_mode: bool = BGR_MODE_DEFAULT,
) -> tuple[np.ndarray, bool]:
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 3 — Знизити пороги детектора тіней

**Файл:** `processing/shadow_remove.py`

**Що зробити:** Замінити константи:

```python
# Було:
SHADOW_DETECT_THRESHOLD = 80   →   SHADOW_DETECT_THRESHOLD = 100
SHADOW_RATIO_THRESHOLD = 0.3   →   SHADOW_RATIO_THRESHOLD = 0.45
```

Пояснення: вищий `threshold` (100 замість 80) означає що детектор
спрацює навіть коли тіні не дуже темні (p5 < 100 замість p5 < 80).
Вищий `ratio` (0.45 замість 0.30) означає що менший перепад яскравості
вже вважається тінню.

- [ ] Відмітити виконання в TASK.md

---

## Крок 4 — Додати ключ shadow_bgr_mode у app_settings.py

**Файл:** `config/app_settings.py`

Додати константу:
```python
DEFAULT_SHADOW_BGR_MODE = False
```

У `load()` додати:
```python
"shadow_bgr_mode": cfg.getboolean("processing", "shadow_bgr_mode",
                                   fallback=DEFAULT_SHADOW_BGR_MODE),
```

У `save()` у секцію `cfg["processing"]` додати:
```python
cfg["processing"]["shadow_bgr_mode"] = str(
    settings.get("shadow_bgr_mode", DEFAULT_SHADOW_BGR_MODE)
).lower()
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 5 — Додати перемикач BGR-режиму у settings_window.py

**Файл:** `gui/settings_window.py`

У метод `_page_shadow_remove()` після існуючих полів додати:
```python
self._cb_shadow_bgr_mode = QCheckBox()
form.addRow("BGR-алгоритм (краще для деяких документів):",
            self._cb_shadow_bgr_mode)
```

У `_apply_settings()` додати:
```python
self._cb_shadow_bgr_mode.setChecked(s.get("shadow_bgr_mode", False))
```

У `_collect_settings()` додати:
```python
"shadow_bgr_mode": self._cb_shadow_bgr_mode.isChecked(),
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 6 — Перевірка

Після всіх змін перевірити:
1. Завантажити будь-яке зображення (включаючи `photo`)
2. Встановити комбобокс "Завжди"
3. Натиснути Auto Fix
4. Переконатися що в лозі з'явився рядок "тіні видалено (примусово)"
5. Перемкнути "BGR-алгоритм" у Налаштуваннях → повторити → порівняти результати

- [ ] Відмітити виконання в TASK.md