## Архітектурне рішення перед планом

**`detect_face()` — окрема функція у `diagnostics.py`, lazy виклик у pipeline.**

Чому так:
- `diagnose()` запускається на кожному зображенні — додавати face detection в `DiagnosticResult` означає +50-100ms на кожне фото навіть якщо shadow_remove взагалі не потрібен
- Lazy виклик: face detection запускається **тільки** коли `uniformity > high` і `auto` режим — вузька умова
- Функція в `diagnostics.py` — одне місце, легко знайти, перевикористати в майбутньому якщо треба

---

# TASK: shadow_remove після perspective + doc_type guard + face detection

## Загальна картина змін

| Файл | Що міняємо |
|---|---|
| `processing/diagnostics.py` | Додаємо функцію `detect_face()` |
| `processing/pipeline.py` | Порядок кроків, видалення дублю, doc_type guard, face detection |
| `gui/settings_window.py` | Порядок у `PIPELINE_STEPS_FIXED_ORDER` |
| `config/app_settings.py` | `DEFAULT_PIPELINE_STEPS_ENABLED` |

---

## Крок 1 — `processing/diagnostics.py`: додати функцію `detect_face()`

Додати **після** функції `measure_background_metrics()` і **перед** функцією `diagnose()` нову standalone функцію:

```python
# Константа для детекції обличчя
FACE_DETECT_MAX_DIM = 400   # зменшуємо для швидкості (~50ms)
FACE_DETECT_MIN_NEIGHBORS = 3
FACE_DETECT_MIN_SIZE = (30, 30)
FACE_DETECT_SCALE_FACTOR = 1.1


def detect_face(image: np.ndarray) -> bool:
    """
    Швидка детекція обличчя через OpenCV Haar cascade.
    Повертає True якщо знайдено хоча б одне обличчя.

    Використовується як захист від auto shadow_remove на документах
    з портретним фото (паспорт, посвідчення, студентський).
    Запускається lazy — тільки коли uniformity висока і shadow_remove
    взагалі міг би спрацювати.

    Зменшує зображення до FACE_DETECT_MAX_DIM по більшій стороні.
    Якщо cascade не завантажився — повертає False (не блокує pipeline).
    """
    h, w = image.shape[:2]
    scale = min(FACE_DETECT_MAX_DIM / max(h, w), 1.0)
    if scale < 1.0:
        small = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        return False

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECT_SCALE_FACTOR,
        minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
        minSize=FACE_DETECT_MIN_SIZE,
    )
    return len(faces) > 0
```

- [x] Відмітити виконання

---

## Крок 2 — `processing/pipeline.py`: змінити порядок у `PIPELINE_STEPS_FIXED_ORDER`

Знайти константу `PIPELINE_STEPS_FIXED_ORDER` у верхній частині файлу.

**Було:**
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

**Стало:**
```python
PIPELINE_STEPS_FIXED_ORDER = [
    ("perspective",      "Авто-перспектива"),
    ("shadow_remove",    "Видалення тіней"),
    ("brightness",       "Авто-яскравість"),
    ("contrast",         "Авто-контраст"),
    ("hdr",              "HDR"),
    ("sharpen",          "Різкість"),
    ("grayscale",        "Grayscale / бінаризація"),
    ("white_background", "Білий фон"),
]
```

- [x] Відмітити виконання

---

## Крок 3 — `processing/pipeline.py`: оновити `_preset_steps_map` всередині `run_autofix()`

**Було:**
```python
_preset_steps_map = {
    "doc_bw":    ["shadow_remove", "perspective", "brightness", "contrast", "sharpen", "grayscale", "white_background"],
    "doc_color": ["shadow_remove", "perspective", "brightness", "contrast", "sharpen", "white_background"],
    "photo":     ["perspective", "hdr", "sharpen"],
    "geometry":  ["perspective"],
}
```

**Стало:**
```python
_preset_steps_map = {
    "doc_bw":    ["perspective", "shadow_remove", "brightness", "contrast", "sharpen", "grayscale", "white_background"],
    "doc_color": ["perspective", "shadow_remove", "brightness", "contrast", "sharpen", "white_background"],
    "photo":     ["perspective", "hdr", "sharpen"],
    "geometry":  ["perspective"],
}
```

- [x] Відмітити виконання

---

## Крок 4 — `processing/pipeline.py`: видалити `_shadow_was_applied` і весь дублюючий другий прохід

### 4a. Видалити ініціалізацію прапорця

Знайти і видалити рядок:
```python
_shadow_was_applied = False
```

### 4b. Видалити всі присвоєння `_shadow_was_applied = True`

У блоці `shadow_remove`, режим `"always"` — видалити:
```python
_shadow_was_applied = True
```

У блоці `shadow_remove`, режим `"auto"`, після `if _should_run:` — видалити:
```python
_shadow_was_applied = True
```

### 4c. Видалити весь блок другого проходу всередині `elif step_key == "perspective":`

Знайти і повністю видалити цей блок (після рядка `_bg_uniformity, _detail_density = _diag.measure_background_metrics(result)`):

```python
# Другий прохід shadow_remove після перспективи (якщо ще не застосовувався)
if not _shadow_was_applied and _bg_uniformity > _shadow_unif_high:
    result, had_shadow = shadow_remove.auto_remove_shadow(
        result,
        is_color_document=_shadow_is_color,
        coarse_blend=_shadow_coarse_blend,
        detect_threshold=_shadow_detect_threshold,
        detect_ratio=_shadow_detect_ratio,
        bgr_mode=_shadow_bgr_mode,
    )
    if had_shadow:
        log_entries.append({"step": "shadow_remove", "applied": True,
                           "detail": f"тіні видалено (після перспективи, uniformity={_bg_uniformity:.2f})"})
    # Нейтралізація кольорового відтінку для phone_camera
    if had_shadow and _capture_cond == CAPTURE_PHONE:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        bg_mask = l_ch > 180
        bg_pixel_count = np.count_nonzero(bg_mask)
        if bg_pixel_count > 0:
            a_bg = a_ch[bg_mask]
            b_bg = b_ch[bg_mask]
            a_shift = (128.0 - float(np.median(a_bg))) * 0.3
            b_shift = (128.0 - float(np.median(b_bg))) * 0.3
            a_ch = np.clip(a_ch.astype(np.float32) + a_shift, 0, 255).astype(np.uint8)
            b_ch = np.clip(b_ch.astype(np.float32) + b_shift, 0, 255).astype(np.uint8)
            merged = cv2.merge([l_ch, a_ch, b_ch])
            result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            log_entries.append({"step": "color_neutralize", "applied": True,
                               "detail": "нейтралізація відтінку (після перспективи)"})
```

- [x] Відмітити виконання

---

## Крок 5 — `processing/pipeline.py`: переписати логіку `auto` режиму в блоці `shadow_remove`

Знайти всередині `elif step_key == "shadow_remove":` блок `else:  # auto`.

**Було:**
```python
else:  # auto — з урахуванням background_uniformity та capture_conditions
    _should_run = False
    _reason = ""
    # screen_capture — не запускаємо shadow_remove (екран уже рівномірний)
    if _capture_cond == CAPTURE_SCREEN:
        _should_run = False
        _reason = f"screen_capture"
    elif _bg_uniformity > _shadow_unif_high:
        # Однорідний фон — запускаємо незалежно від doc_type
        _should_run = True
        _reason = f"uniformity={_bg_uniformity:.2f}>{_shadow_unif_high:.2f}"
    elif _bg_uniformity < _shadow_unif_low:
        # Складний фон — не запускаємо
        _should_run = False
        _reason = f"uniformity={_bg_uniformity:.2f}<{_shadow_unif_low:.2f}"
    else:
        # Проміжний діапазон — залишаємо стару логіку (тільки для документів)
        if doc_type == DocType.BW_DOCUMENT.value:
            _should_run = True
            _reason = f"doc_type={doc_type}"
```

**Стало:**
```python
else:  # auto — з урахуванням background_uniformity, doc_type та capture_conditions
    _should_run = False
    _reason = ""

    if _capture_cond == CAPTURE_SCREEN:
        # screen_capture — не запускаємо (екран рівномірний, не тінь)
        _should_run = False
        _reason = "screen_capture"
    elif doc_type in (DocType.COLOR_DOCUMENT.value, DocType.PHOTO.value):
        # Кольорові документи та фото — не запускаємо в auto.
        # Паспорти, посвідчення, фотографії псуються shadow_remove.
        # Користувач може примусово увімкнути через режим "always".
        _should_run = False
        _reason = f"doc_type={doc_type} (auto skip)"
    elif _bg_uniformity > _shadow_unif_high:
        # Однорідний фон bw_document / flat_background.
        # Додаткова перевірка: якщо є обличчя — це документ з портретом,
        # shadow_remove зіпсує фото особи.
        from processing import diagnostics as _diag_face
        if _diag_face.detect_face(result):
            _should_run = False
            _reason = f"face_detected (uniformity={_bg_uniformity:.2f})"
        else:
            _should_run = True
            _reason = f"uniformity={_bg_uniformity:.2f}>{_shadow_unif_high:.2f}"
    elif _bg_uniformity < _shadow_unif_low:
        # Складний фон — не запускаємо
        _should_run = False
        _reason = f"uniformity={_bg_uniformity:.2f}<{_shadow_unif_low:.2f}"
    else:
        # Проміжний діапазон — тільки для bw_document,
        # і тільки якщо немає обличчя
        if doc_type == DocType.BW_DOCUMENT.value:
            from processing import diagnostics as _diag_face
            if _diag_face.detect_face(result):
                _should_run = False
                _reason = "face_detected (mid uniformity)"
            else:
                _should_run = True
                _reason = f"doc_type={doc_type}"
```

> ⚠️ `detect_face(result)` — викликається на `result` (поточний стан зображення після perspective), не на оригінальному `image`. Це важливо — обличчя шукаємо на вже вирівняному зображенні.

- [x] Відмітити виконання

---

## Крок 6 — `processing/pipeline.py`: переконатися що у `perspective`-блоці залишилось оновлення uniformity

Після видалення другого проходу (Крок 4c) блок `elif step_key == "perspective":` має виглядати так:

```python
elif step_key == "perspective":
    _use_persp = use_perspective or _capture_cond == CAPTURE_SCREEN
    if _use_persp:
        result, persp_status = run_perspective_auto_smart(result, settings)
        # Оновлюємо uniformity після виправлення перспективи —
        # shadow_remove іде наступним і побачить актуальне значення
        _bg_uniformity, _detail_density = _diag.measure_background_metrics(result)
        if persp_status not in ("перспектива не потрібна", "перспектива не потребує корекції"):
            log_entries.append({"step": "perspective", "applied": True, "detail": persp_status})
        elif _capture_cond == CAPTURE_SCREEN:
            log_entries.append({"step": "perspective_forced", "applied": True, "detail": "screen_capture detected"})
```

Рядок `_bg_uniformity, _detail_density = _diag.measure_background_metrics(result)` — обов'язково залишити. Нічого більше у цьому блоці не має бути.

- [x] Відмітити виконання

---

## Крок 7 — `gui/settings_window.py`: оновити `PIPELINE_STEPS_FIXED_ORDER`

**Було:**
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

**Стало:**
```python
PIPELINE_STEPS_FIXED_ORDER = [
    ("perspective",      "Авто-перспектива"),
    ("shadow_remove",    "Видалення тіней"),
    ("brightness",       "Авто-яскравість"),
    ("contrast",         "Авто-контраст"),
    ("hdr",              "HDR"),
    ("sharpen",          "Різкість"),
    ("grayscale",        "Grayscale / бінаризація"),
    ("white_background", "Білий фон"),
]
```

- [x] Відмітити виконання

---

## Крок 8 — `config/app_settings.py`: оновити `DEFAULT_PIPELINE_STEPS_ENABLED`

**Було:**
```python
DEFAULT_PIPELINE_STEPS_ENABLED = "shadow_remove,perspective,brightness,contrast,hdr,sharpen,grayscale,white_background"
```

**Стало:**
```python
DEFAULT_PIPELINE_STEPS_ENABLED = "perspective,shadow_remove,brightness,contrast,hdr,sharpen,grayscale,white_background"
```

- [x] Відмітити виконання

---

## Крок 9 — Фінальна перевірка агента

Переконатись після всіх змін:

1. `PIPELINE_STEPS_FIXED_ORDER` в `pipeline.py` та `settings_window.py` — однакові, `perspective` перший, `shadow_remove` другий
2. У `run_autofix()` немає жодного `_shadow_was_applied`
3. Блок `elif step_key == "perspective":` — не містить жодних викликів `shadow_remove` або `auto_remove_shadow`
4. Блок `elif step_key == "shadow_remove":`, режим `"auto"` — перевірка `doc_type in (DocType.COLOR_DOCUMENT.value, DocType.PHOTO.value)` стоїть **перед** перевіркою uniformity
5. `detect_face(result)` викликається **тільки** у двох місцях: `_bg_uniformity > _shadow_unif_high` та проміжний діапазон для `bw_document`
6. `_bg_uniformity, _detail_density = _diag.measure_background_metrics(result)` — присутній у `perspective`-блоці після `run_perspective_auto_smart()`
7. Функція `detect_face()` у `diagnostics.py` — повертає `False` якщо cascade не завантажився (не ламає pipeline)

- [x] Відмітити виконання