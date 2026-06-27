на кольоровому документі фон жовтого відтінку(чи будь який інший відтінок) і фон не видаляється це потрібно якось вирішити


# План рефакторингу перспективи — Варіант B+C

---

## Архітектурне рішення

### Ключові інваріанти нової системи

```
_base_for_perspective  = знімок ДО перспективи (для undo/scaling)
_perspective_corners   = поточні кути у просторі _base_for_perspective
_base                  = НЕ змінюється під час perspective-edit
_processed             = perspective(_base_for_perspective, corners) + sliders

BEFORE панель → завжди показує make_preview(_base_for_perspective)
AFTER панель  → завжди показує _processed (live результат)
Кути (drag)   → на BEFORE панелі, координати в просторі preview(_base_for_perspective)
```

**Чому кути залишаються на BEFORE:** стандартний UX (як Lightroom/PS) — виставляєш кути на джерелі, бачиш результат справа. AFTER показує результат **без** кутів — чистий live preview.

---

## Phase 1 — Прибирання кнопки підтвердження та стан

### 1.1 Нові атрибути в `MainWindow.__init__`

```
_base_for_perspective  → вже є  
_perspective_corners   → np.ndarray | None  ← НОВИЙ (замість логіки в _on_persp_pts)
```

- [ ] Відмітити виконання

### 1.2 Видалення кнопки з `gui/main_window.py`

Видалити повністю:
- `self._btn_apply_perspective = QPushButton(...)`
- `self._btn_apply_perspective.clicked.connect(self._do_apply_perspective)`
- `buttons_row.addWidget(self._btn_apply_perspective)`
- метод `_do_apply_perspective(self)`

В `_update_buttons` видалити рядки:
```python
has_pending_persp = self._base_for_perspective is not None
self._btn_apply_perspective.setVisible(has_pending_persp)
```

- [ ] Відмітити виконання

### 1.3 Авто-commit без кнопки

Perspective-результат завжди в `_processed`. Commit `_processed → _base` відбувається в:

| Момент | Дія |
|---|---|
| `_do_print_current` | вже є — залишити |
| `_do_autofix_classic` | додати commit перед roботою |
| `_load_next_manual` | додати commit перед переходом |
| `_do_reset_all` | очистити corners |
| `_clear_queue` | очистити corners |

У кожному з перелічених місць додати блок:
```python
if self._base_for_perspective is not None and self._perspective_corners is not None:
    self._base = pipeline.run_perspective_manual(
        self._base_for_perspective, self._perspective_corners
    )
    self._base_for_perspective = None
    self._perspective_corners = None
    self._preview.disable_perspective_edit()
```

- [ ] Відмітити виконання

---

## Phase 2 — Виправлення системи координат

### 2.1 Єдина функція масштабування (додати в `main_window.py`)

```python
def _corners_to_preview_pts(
    self,
    corners: np.ndarray,
    source: np.ndarray,
) -> list[QPoint]:
    """Масштабує кути з простору source у простір make_preview(source)."""
    prev = image_utils.make_preview(source)
    prev_h, prev_w = prev.shape[:2]
    src_h, src_w = source.shape[:2]
    sx = prev_w / max(src_w, 1)
    sy = prev_h / max(src_h, 1)
    return [QPoint(int(c[0] * sx), int(c[1] * sy)) for c in corners]

def _preview_pts_to_corners(
    self,
    points: list[QPoint],
    source: np.ndarray,
) -> np.ndarray:
    """Масштабує точки з простору make_preview(source) у простір source."""
    prev = image_utils.make_preview(source)
    prev_h, prev_w = prev.shape[:2]
    src_h, src_w = source.shape[:2]
    sx = src_w / max(prev_w, 1)
    sy = src_h / max(prev_h, 1)
    return np.array(
        [[p.x() * sx, p.y() * sy] for p in points],
        dtype=np.float32,
    )
```

- [ ] Відмітити виконання

### 2.2 Виправлення `_show_perspective_points`

**Поточний баг:** використовує `self._base` (може бути вже трансформованим).

**Новий код:**
```python
def _show_perspective_points(
    self, corners: np.ndarray, status_msg: str
) -> None:
    """corners — у просторі _base_for_perspective."""
    if self._base_for_perspective is None:
        return
    # Оновлюємо BEFORE: показуємо джерело (до перспективи)
    prev_source = image_utils.make_preview(self._base_for_perspective)
    self._preview.set_before(prev_source)
    # Масштабуємо кути
    pts = self._corners_to_preview_pts(corners, self._base_for_perspective)
    self._preview.enable_perspective_edit(pts)
    self._set_status(status_msg)
```

- [ ] Відмітити виконання

### 2.3 Виправлення `_on_persp_pts`

**Поточний баг:** масштаб рахується через `self._base`, а не `_base_for_perspective`.

**Новий код:**
```python
def _on_persp_pts(self, points: list) -> None:
    if self._base_for_perspective is None or len(points) != 4:
        return
    try:
        corners = self._preview_pts_to_corners(
            points, self._base_for_perspective
        )
        self._perspective_corners = corners
        persp_result = pipeline.run_perspective_manual(
            self._base_for_perspective, corners
        )
        # Накладаємо слайдери
        vals = self._controls.values()
        result = pipeline.run_manual_adjustments(
            persp_result,
            brightness=vals["brightness"],
            contrast=vals["contrast"],
            sharpen_strength=vals["sharpen_strength"],
            hdr_strength=vals["hdr_strength"],
            grayscale=vals["grayscale"],
            shadow_highlight_strength=vals["shadow_highlight"],
            contrast_mode=self._settings.get("contrast_mode", "linear"),
        )
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
    except Exception as e:
        self._logger.error(f"Помилка перспективи: {e}", exc_info=True)
        self._set_status(f"Помилка: {e}")
```

- [ ] Відмітити виконання

---

## Phase 3 — Виправлення `_on_controls_changed` для perspective-режиму

**Поточна проблема:** слайдери застосовуються до `_base` (до перспективи), а не до результату перспективи.

```python
def _on_controls_changed(self, vals: dict = None) -> None:
    if self._base is None and self._base_for_perspective is None:
        return
    try:
        vals = self._controls.values()
        
        # Визначаємо базу з урахуванням активної перспективи
        if (self._base_for_perspective is not None
                and self._perspective_corners is not None):
            base_for_sliders = pipeline.run_perspective_manual(
                self._base_for_perspective, self._perspective_corners
            )
        else:
            base_for_sliders = self._base
        
        result = pipeline.run_manual_adjustments(
            base_for_sliders,
            brightness=vals["brightness"],
            contrast=vals["contrast"],
            sharpen_strength=vals["sharpen_strength"],
            hdr_strength=vals["hdr_strength"],
            grayscale=vals["grayscale"],
            shadow_highlight_strength=vals["shadow_highlight"],
            contrast_mode=self._settings.get("contrast_mode", "linear"),
        )
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
        self._preview.set_autofix_applied(None)
        self._update_buttons()
        self._store_current_settings()
    except Exception as e:
        self._logger.error(f"Помилка слайдерів: {e}", exc_info=True)
```

> ⚠️ `run_perspective_manual` на великому зображенні при кожному русі слайдера — може бути повільним. Кешувати результат перспективи в `_perspective_result_cache` (очищати в `_on_persp_pts`).

- [ ] Відмітити виконання

---

## Phase 4 — Виправлення `_do_persp_manual`

### 4.1 Default corners — не кути кадру, а 80% центр

```python
def _default_perspective_corners(self, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    mx = int(w * 0.10)
    my = int(h * 0.10)
    return np.array(
        [[mx, my], [w-mx, my], [w-mx, h-my], [mx, h-my]],
        dtype=np.float32,
    )
```

### 4.2 Новий `_do_persp_manual`

```python
def _do_persp_manual(self) -> None:
    if self._base is None:
        return
    # Commit попередньої перспективи якщо була
    if self._base_for_perspective is not None and self._perspective_corners is not None:
        self._base = pipeline.run_perspective_manual(
            self._base_for_perspective, self._perspective_corners
        )
    
    # Зберігаємо знімок джерела
    self._base_for_perspective = self._base.copy()
    self._preview.disable_perspective_edit()
    
    # Спробуємо детектувати кути
    corners = pipeline.detect_corners(self._base_for_perspective)
    if corners is None:
        corners = self._default_perspective_corners(self._base_for_perspective)
        status = "Встановіть кути вручну (документ не знайдено)"
    else:
        status = "Тягніть кути для корекції перспективи"
    
    # Зберігаємо кути у просторі зображення
    self._perspective_corners = corners
    
    # Оновлюємо BEFORE: показуємо джерело
    prev_source = image_utils.make_preview(self._base_for_perspective)
    self._preview.set_before(prev_source)
    
    # Показуємо live результат на AFTER
    try:
        persp_result = pipeline.run_perspective_manual(
            self._base_for_perspective, corners
        )
        self._processed = persp_result
        self._preview.set_after(image_utils.make_preview(persp_result))
    except Exception:
        self._preview.set_after(prev_source)
    
    # Виставляємо точки на BEFORE
    pts = self._corners_to_preview_pts(corners, self._base_for_perspective)
    self._preview.enable_perspective_edit(pts)
    self._set_status(status)
    self._update_buttons()
```

- [ ] Відмітити виконання

---

## Phase 5 — Виправлення `_do_persp_auto`

**Поточний баг:** після `apply_correction` → `_base` змінює розміри → `corners_before` некоректні.

```python
def _do_persp_auto(self) -> None:
    if self._base is None:
        return
    if self._single_thread is not None and self._single_thread.isRunning():
        self._set_status("⏳ Зачекайте…")
        return
    
    # Зберігаємо знімок джерела ДО будь-яких змін
    self._base_for_perspective = self._base.copy()
    source_snapshot = self._base_for_perspective  # alias для closure

    def _work():
        corners = pipeline.detect_corners(source_snapshot)
        if corners is None:
            # Fallback: тільки deskew
            from processing import deskew as deskew_module
            angle = deskew_module.measure_skew_angle(source_snapshot)
            if abs(angle) >= deskew_module.DESKEW_MIN_ANGLE:
                result = deskew_module.apply_deskew(source_snapshot, angle)
                return "deskew", None, result
            return "noop", None, source_snapshot.copy()
        
        from processing.perspective import detect_skewed_sides, _order_points
        ordered = _order_points(corners)
        skewed = detect_skewed_sides(ordered)
        if not any(skewed.values()):
            return "straight", corners, source_snapshot.copy()
        
        # Є перспективне викривлення
        result = pipeline.run_perspective_manual(source_snapshot, corners)
        return "corrected", corners, result

    def _on_done(payload):
        action, corners, result = payload
        
        if action == "noop":
            self._base_for_perspective = None
            self._perspective_corners = None
            self._set_status("Перспектива не потрібна")
        
        elif action == "deskew":
            self._base = result.copy()
            self._base_for_perspective = None
            self._perspective_corners = None
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            self._set_status("Нахил виправлено (deskew)")
        
        elif action == "straight":
            # Немає викривлення, але показуємо кути для тонкого налаштування
            self._perspective_corners = corners
            prev_source = image_utils.make_preview(self._base_for_perspective)
            self._preview.set_before(prev_source)
            pts = self._corners_to_preview_pts(corners, self._base_for_perspective)
            self._preview.enable_perspective_edit(pts)
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            self._set_status("Документ рівний — підправте кути за потреби")
        
        elif action == "corrected":
            # Перспектива виправлена — входимо в режим редагування
            self._perspective_corners = corners
            prev_source = image_utils.make_preview(self._base_for_perspective)
            self._preview.set_before(prev_source)
            pts = self._corners_to_preview_pts(corners, self._base_for_perspective)
            self._preview.enable_perspective_edit(pts)
            self._processed = result
            self._preview.set_after(image_utils.make_preview(result))
            self._set_status("Перспективу виправлено — підправте кути за потреби")
        
        self._update_buttons()
    
    self._run_in_background(_work, _on_done, button_to_lock=self._btn_autofix)
```

- [ ] Відмітити виконання

---

## Phase 6 — `_do_persp_reset` без кнопки підтвердження

```python
def _do_persp_reset(self) -> None:
    if self._orig is None:
        return
    # Повертаємось до стану до входу в perspective-режим
    if self._base_for_perspective is not None:
        self._base = self._base_for_perspective.copy()
    else:
        self._base = self._orig.copy()
    
    self._base_for_perspective = None
    self._perspective_corners = None
    self._processed = self._base.copy()
    self._preview.set_before(image_utils.make_preview(self._orig))
    self._preview.set_after(image_utils.make_preview(self._base))
    self._preview.disable_perspective_edit()
    self._set_status("Перспективу скинуто")
    self._update_buttons()
    self._on_controls_changed()
```

- [ ] Відмітити виконання

---

## Phase 7 — Scoring детекції в `perspective.py`

### 7.1 Нова функція `_score_quad`

```python
def _score_quad(pts: np.ndarray, image_shape: tuple) -> float:
    """
    Скоринг знайденого квадрилатераля: 0.0 (поганий) → 1.0 (ідеальний).
    Критерії: площа, кути ~90°, відступ від краю.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)
    ordered = _order_points(pts.astype(np.float32))
    tl, tr, br, bl = ordered
    
    # Площа: 15–85% кадру = ідеально
    area = float(cv2.contourArea(ordered))
    ratio = area / image_area
    if ratio < 0.10 or ratio > 0.92:
        return 0.0
    # Пік скору на ratio ≈ 0.50
    area_score = 1.0 - abs(ratio - 0.50) / 0.42
    
    # Кути між сторонами: ближче до 90° = краще
    sides = [tr-tl, br-tr, bl-br, tl-bl]
    angle_devs = []
    for i in range(4):
        v1 = sides[i].astype(np.float64)
        v2 = sides[(i+1) % 4].astype(np.float64)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1.0 or n2 < 1.0:
            return 0.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_a)))
        angle_devs.append(abs(angle - 90.0))
    angle_score = 1.0 - float(np.mean(angle_devs)) / 45.0
    angle_score = max(0.0, angle_score)
    
    # Відступ від краю: кути не мають торкатися кадру
    margin = 0.04
    pts_xs = [tl[0], tr[0], br[0], bl[0]]
    pts_ys = [tl[1], tr[1], br[1], bl[1]]
    touches_edge = (
        min(pts_xs) < w * margin or max(pts_xs) > w * (1 - margin) or
        min(pts_ys) < h * margin or max(pts_ys) > h * (1 - margin)
    )
    margin_score = 0.6 if touches_edge else 1.0
    
    return area_score * 0.45 + angle_score * 0.45 + margin_score * 0.10
```

- [ ] Відмітити виконання

### 7.2 Переписати `_detect_corners_impl` — зібрати всі кандидати, повернути найкращого

```python
def _detect_corners_impl(gray: np.ndarray) -> np.ndarray | None:
    clahe_gray = _apply_clahe(gray)
    candidates = []  # list[tuple[score, corners]]

    methods = [
        ("adaptive",        lambda g: _try_adaptive_threshold(g)),
        ("canny",           lambda g: _try_canny(g)),
        ("clahe+adaptive",  lambda g: _try_adaptive_threshold(clahe_gray)),
        ("clahe+canny",     lambda g: _try_canny(clahe_gray)),
        ("hough",           lambda g: _try_hough_lines(g)),
    ]

    for name, fn in methods:
        corners = fn(gray)
        if corners is not None:
            score = _score_quad(corners, gray.shape)
            if score > 0.0:
                refined = _refine_corners_subpix(gray, corners)
                candidates.append((score, refined))
                logger.debug(f"_detect_corners_impl: {name} score={score:.3f}")

    if not candidates:
        # Fallback: largest contour
        corners = _try_largest_contour(gray)
        if corners is not None:
            refined = _refine_corners_subpix(gray, corners)
            return refined
        return None

    # Повертаємо кандидата з найвищим скором
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_corners = candidates[0]
    logger.debug(f"_detect_corners_impl: вибрано score={best_score:.3f} з {len(candidates)} кандидатів")
    return best_corners
```

- [ ] Відмітити виконання

---

## Phase 8 — Виправлення `_update_buttons`

Прибрати все пов'язане з `_btn_apply_perspective`, додати стан perspective-режиму:

```python
def _update_buttons(self) -> None:
    has_queue  = bool(self._queue.get_all_paths())
    has_img    = self._orig is not None
    is_batch   = self._radio_auto.isChecked()
    in_persp   = self._base_for_perspective is not None

    self._btn_print_all.setEnabled(has_queue and is_batch)
    self._btn_print.setEnabled(has_img)
    self._btn_skip.setEnabled(
        self._processor.has_next() and not is_batch
    )
    # Авто Фікс доступний і під час редагування перспективи
    self._btn_autofix.setEnabled(has_img)
    self._btn_save_img.setEnabled(has_img)
    # Кнопка скидання перспективи — тільки в perspective-режимі
    # (вона вже є в controls, не потребує додаткового управління)
```

- [ ] Відмітити виконання

---

## Порядок виконання агентом

```
Phase 7 → perspective.py (незалежний модуль, без GUI)
Phase 1 → main_window.py (видалення кнопки)
Phase 2 → main_window.py (helper-функції масштабування)
Phase 3 → main_window.py (_on_controls_changed)
Phase 4 → main_window.py (_do_persp_manual)
Phase 5 → main_window.py (_do_persp_auto)
Phase 6 → main_window.py (_do_persp_reset)
Phase 8 → main_window.py (_update_buttons)
```

> ⚠️ `run_perspective_manual` в `_on_controls_changed` при кожному русі слайдера може гальмувати на великих зображеннях (>8МП). Якщо виникне проблема — додати `_perspective_cached_result` (інвалідувати в `_on_persp_pts`).