# PHOTOPRINT: Повний план виправлень

---

## БЛОК 1: Критичні баги

### 1.1 — Кеш прев'ю за адресою пам'яті (image_utils.py)
**Проблема:** `image.ctypes.data` — адреса пам'яті. Новий масив може зайняти ту саму адресу → кеш повертає стале зображення попереднього файлу.

**Файл:** `utils/image_utils.py` → `_preview_cache_key`

**Кроки:**
- [x] Замінити `image.ctypes.data` на `hash(image.tobytes()[:1024])` — хеш першого 1KB пікселів (баланс між унікальністю та швидкістю)
- [x] Додати до ключа `image.shape` і `image.dtype` — залишити як є
- [ ] Протестувати: завантажити файл A → перейти на B → повернутись на A → прев'ю має показувати A, не B
- [x] Відмітити виконання.

---

### 1.2 — `shadow_remove_mode` не зберігається (app_settings.py)
**Проблема:** У функції `save()` відсутній рядок збереження `shadow_remove_mode`. При перезапуску завжди "авто".

**Файл:** `config/app_settings.py`

**Кроки:**
- [x] У функції `save()` у секції `cfg["processing"]` додати:
  `"shadow_remove_mode": settings.get("shadow_remove_mode", "auto")`
- [x] У функції `load()` додати:
  `"shadow_remove_mode": cfg.get("processing", "shadow_remove_mode", fallback="auto")`
- [ ] Перевірити: змінити режим на "завжди" → перезапустити → режим має зберегтись
- [x] Відмітити виконання.

---

### 1.3 — `_collect_settings` не збирає всі параметри (settings_window.py)
**Проблема:** `shadow_uniform_std_threshold` і `shadow_uniform_block_size` — є в `app_settings.py`, але відсутні в `_collect_settings()` у `SettingsWindow`. Зміни через UI не зберігаються.

**Файл:** `gui/settings_window.py`

**Кроки:**
- [x] У `_page_shadow_remove()` додати два `QDoubleSpinBox`/`QSpinBox`:
  - `self._spin_uniform_std_threshold` (range 0.5–20.0, step 0.5, decimals 1)
  - `self._spin_uniform_block_size` (range 8–64, step 8)
  - Підписи: "Поріг std для детекції тіні на рівному фоні:" і "Розмір блоку аналізу (px):"
- [x] У `_apply_settings()` встановити значення з `s`:
  `self._spin_uniform_std_threshold.setValue(s.get("shadow_uniform_std_threshold", 2.0))`
  `self._spin_uniform_block_size.setValue(s.get("shadow_uniform_block_size", 32))`
- [x] У `_collect_settings()` додати:
  `"shadow_uniform_std_threshold": self._spin_uniform_std_threshold.value()`
  `"shadow_uniform_block_size": int(self._spin_uniform_block_size.value())`
- [x] Відмітити виконання.

---

### 1.4 — `reset()` слайдера завжди до 0.0 замість default (controls.py)
**Проблема:** `_SliderRow.reset()` викликає `set_value(0.0)` — але різкість має default 0.4, тіні 0.0, HDR 0.0. Після "Скинути слайдери" різкість стає 0.

**Файл:** `gui/controls.py`

**Кроки:**
- [x] У `_SliderRow.__init__` зберігати default: `self._default = default`
- [x] У `_SliderRow.reset()` замінити `self.set_value(0.0)` на `self.set_value(self._default, silent=True)`
- [x] У `ControlsPanel.reset_all()` переконатись що всі 5 слайдерів мають правильні defaults при ініціалізації
- [ ] Перевірити: виставити різкість 1.0 → "Скинути слайдери" → різкість має стати 0.4
- [x] Відмітити виконання.

---

### 1.5 — Подвійна обробка для `bw_document` (pipeline.py)
**Проблема:** Крок `sharpen` викликає `autofix.apply_bw_document()` яка робить `auto_contrast + to_grayscale`. Потім pipeline окремо запускає кроки `contrast` і `grayscale` → подвійна обробка.

**Файл:** `processing/pipeline.py` → функція `run_autofix`

**Кроки:**
- [x] У кроці `sharpen` для `bw_document`: замінити виклик `autofix.apply_bw_document()` на прямий виклик лише `sharpen.apply(result, strength=sharpen_strength)` — без auto_contrast і grayscale всередині
- [x] Переконатись що кроки `contrast` і `grayscale` в PIPELINE_STEPS_FIXED_ORDER стоять ПІСЛЯ `sharpen` і виконуються окремо
- [x] У `autofix.apply_bw_document()` додати параметр `skip_contrast=False` і `skip_grayscale=False` для зворотньої сумісності
- [ ] Перевірити на чорно-білому документі: текст не має бути "зайво темним" або "двічі контрастним"
- [x] Відмітити виконання.

---

### 1.6 — `detect_face` викликається до 3 разів підряд (pipeline.py)
**Проблема:** У різних гілках `shadow_remove` → `auto` детекція обличчя запускається незалежно для `photo`, `color_document` і `bw_document`. Кожен виклик ~50мс.

**Файл:** `processing/pipeline.py` → `run_autofix`

**Кроки:**
- [x] Додати змінну `_face_detected: bool | None = None` перед циклом по кроках
- [x] Зробити виклик один раз на початку (не для PHOTO), зберегти результат
- [x] Замінити всі прямі виклики `_diag.detect_face(result)` у гілках shadow_remove на `_face_detected`
- [x] Відмітити виконання.

---

## БЛОК 2: Архітектурні проблеми

### 2.1 — Debounce для слайдерів (main_window.py)
**Проблема:** Кожен піксель руху слайдера → повна обробка зображення синхронно в GUI-потоці. На великих зображеннях UI підвисає.

**Файл:** `gui/main_window.py`

**Кроки:**
- [x] У `MainWindow.__init__` додати `QTimer` з 120мс debounce
- [x] Перейменувати `_on_controls_changed` на `_on_controls_changed_debounced`
- [x] Створити новий `_on_controls_changed` який лише запускає таймер
- [x] Переконатись що при `reset_all` таймер скасовується і обробка запускається негайно
- [x] Відмітити виконання.

---

### 2.2 — `resizeEvent` пише в INI на кожен піксель (main_window.py)
**Проблема:** `resizeEvent` → `_save_window_geometry()` → `app_settings.save()` → запис файлу. При зміні розміру вікна — десятки записів за секунду.

**Файл:** `gui/main_window.py`

**Кроки:**
- [x] Видалити виклик `_save_window_geometry()` з `resizeEvent`
- [x] Додати `QTimer` для відкладеного збереження (1000мс)
- [x] У `resizeEvent` замість прямого збереження: `self._save_geometry_timer.start()`
- [x] У `closeEvent` і `_quit_app` викликати `_save_window_geometry()` безпосередньо
- [x] Відмітити виконання.

---

### 2.3 — `_wait_for_threads` блокує GUI при закритті (main_window.py)
**Проблема:** `self._auto_thread.wait(5000)` блокує GUI-потік → вікно "завмирає" до 5 секунд.

**Файл:** `gui/main_window.py` → `_wait_for_threads`, `closeEvent`

**Кроки:**
- [x] У `closeEvent` замість синхронного очікування: показати `QProgressDialog` "Завершення обробки..."
- [x] Запустити `QTimer.singleShot(100, self._check_threads_and_close)` — перевіряти кожні 100мс
- [x] `_check_threads_and_close`: якщо потоки завершились → закрити; якщо ні → restart таймера; якщо пройшло > 5с → примусово
- [x] Викликати `event.ignore()` одразу у `closeEvent` і закривати програму лише після реального завершення потоків
- [x] Відмітити виконання.

---

### 2.4 — Прогрес під час друку (batch_processor.py, main_window.py)
**Проблема:** Фаза 1 (обробка) показує прогрес до 100%, потім тиша під час фази 2 (друк). Користувач не знає що відбувається.

**Файл:** `batch/batch_processor.py` → `run_auto`, `gui/main_window.py`

**Кроки:**
- [x] У `run_auto` додати окремий callback `on_print_progress: Callable[[int, int, str], None] | None = None`
- [x] У фазі 2 (послідовний друк) викликати `on_print_progress(i+1, total, filename)` перед кожним `print_image`
- [x] У `AutoWorker` додати сигнал `print_progress = pyqtSignal(int, int, str)`
- [x] У `MainWindow` додати `_on_auto_print_progress` — показувати "Друк" vs "Обробка" в статусному рядку
- [x] У progress bar використовувати range 0..total для обробки, потім total+cur для друку
- [x] Відмітити виконання.

---

### 2.5 — Розсинхронізація черги та `_processor` (main_window.py)
**Проблема:** `_processor.set_files()` викликається в різних місцях. Файли додані до `_queue` після останнього `set_files` процесором не обробляються.

**Файл:** `gui/main_window.py`

**Кроки:**
- [x] Видалити всі виклики `_processor.set_files()` і `_processor.add_files()` окрім одного — єдиного місця синхронізації
- [x] Створити метод `_sync_processor_with_queue()`:
- [x] Викликати `_sync_processor_with_queue()` лише у `_do_print_all` і `_start_auto` — безпосередньо перед запуском
- [x] У `_on_files_added` і `_browse_folder` — НЕ оновлювати процесор, тільки чергу
- [x] Відмітити виконання.

---

### 2.6 — `partial_perspective` ігнорується в ітеративній корекції (perspective.py)
**Проблема:** `auto_correct_iterative` завжди використовує `apply_correction` (повна корекція), ігнорує параметр `partial_perspective`.

**Файл:** `processing/perspective.py` → `auto_correct_iterative`

**Кроки:**
- [x] Додати параметр `partial: bool = False` до `auto_correct_iterative`
- [x] Всередині циклу замінити `apply_correction(current, corners)` на: `apply_partial_correction(current, corners) if partial else apply_correction(current, corners)`
- [x] У `pipeline.py` → `run_perspective_auto_smart` передавати `partial=settings.get("partial_perspective", False)` у виклик `auto_correct_iterative`
- [x] Відмітити виконання.

---

## БЛОК 3: Перспектива — повний рефакторинг

### 3.1 — Виправлення формул зворотньої трансформації (perspective.py)
**Проблема:** Неправильні формули у `_detect_with_rotation_candidates` для 90° і 270° → дзеркалення документа, точки поза зображенням.

**Файл:** `processing/perspective.py` → `_detect_with_rotation_candidates`

**Кроки:**
- [x] Гілка `angle == 90`: `back = np.array([[rh - 1 - c[1], c[0]] for c in corners], ...)`
- [x] Гілка `angle == 270`: `back = np.array([[c[1], rw - 1 - c[0]] for c in corners], ...)`
- [x] Гілка `angle == 180`: перевірено — `[w-1-c[0], h-1-c[1]]` (правильно)
- [x] Перед `back`-трансформацією зберігати `rh, rw = rotated.shape[:2]` для кожної ітерації
- [x] Відмітити виконання.

---

### 3.2 — Пріоритет оригінальної орієнтації + перевірка clockwise (perspective.py)
**Проблема:** Ротовані кандидати можуть "перемогти" оригінальну орієнтацію через дрібні відмінності score.

**Файл:** `processing/perspective.py`

**Кроки:**
- [x] Додати функцію `_is_valid_orientation(pts)` (реалізовано як `_is_clockwise`)
- [x] У `_detect_with_rotation_candidates` після back-трансформації: якщо `not _is_clockwise(back)` → `continue`
- [x] Для `angle == 0` множити score на 1.10 (10% бонус)
- [x] Відмітити виконання.

---

### 3.3 — Обмеження початкових точок у межах зображення (main_window.py)
**Проблема:** `detect_corners` може повернути точки за межами зображення → їх виставляють на прев'ю як точки редагування.

**Файл:** `gui/main_window.py`

**Кроки:**
- [x] Додати функцію `_validate_corners_in_bounds(corners, image)` — перевіряє [-20%, 120%]
- [x] У `_do_persp_manual` після виклику `detect_corners`: якщо `not _validate_corners_in_bounds` → fallback до default
- [x] Те саме у `_do_persp_auto` → `_on_done`
- [x] Відмітити виконання.

---

### 3.4 — Стабілізація зображення "До" при редагуванні точок (preview.py, main_window.py)
**Проблема:** `points_changed` (mousemove) → важкий pipeline → зміна розміру "Після" → зсув "До".

**Файл:** `gui/preview.py`, `gui/main_window.py`

**Кроки:**
- [x] У `ImageLabel` додати сигнал `points_released = pyqtSignal(list)` — емітується лише в `mouseReleaseEvent`
- [x] У `PreviewPanel` прокинути `perspective_points_released = pyqtSignal(list)`
- [x] `_preview.perspective_points_changed` → легкий `_on_persp_pts_light`
- [x] `_preview.perspective_points_released` → важкий `_on_persp_pts_heavy`
- [x] Відмітити виконання.

---

### 3.5 — Фіксація розміру панелей під час редагування перспективи (main_window.py)
**Кроки:**
- [x] Додати метод `_freeze_preview_panels()` — `panel.setFixedSize(panel.size())`
- [x] Додати метод `_unfreeze_preview_panels()` — відновлює Expanding політику
- [x] Викликати `_freeze_preview_panels()` у `_do_persp_manual` і `_do_persp_auto`
- [x] Викликати `_unfreeze_preview_panels()` у `_do_persp_reset`
- [x] Відмітити виконання.

---

### 3.6 — Захист від відображення попереднього файлу (main_window.py)
**Кроки:**
- [x] На початку `_on_queue_selection` — зупиняти активний single-thread
- [x] У `_do_autofix_classic` зберігати snapshot поточного шляху: `path_snapshot = self._current_path`
- [x] У `_on_done` замикання перевіряти: `if self._current_path != path_snapshot: return`
- [x] Викликати `image_utils.preview_cache_clear()` на початку `_on_queue_selection`
- [x] Відмітити виконання.

---

## БЛОК 4: UX та логічні проблеми

### 4.1 — Жовтий відтінок фону не видаляється (pipeline.py)
**Проблема:** `color_cast.correct_color_cast` запускається після класифікації і shadow_remove, але відтінок впливає на класифікацію.

**Файл:** `processing/pipeline.py`

**Кроки:**
- [ ] Перемістити перший виклик `color_cast.correct_color_cast` ПЕРЕД `doc_classifier.classify`
- [ ] Зберегти прапор: `result, _had_color_cast = color_cast.correct_color_cast(result)`
- [ ] Для `CAPTURE_PHONE`: знизити `_shadow_unif_high_photo` на 15%
- [ ] Якщо `_had_color_cast == True` і `doc_type == BW_DOCUMENT` і `shadow_mode == "auto"` → встановити `_should_run = True`
- [ ] Відмітити виконання.

---

### 4.2 — Артефакти кольорів у BGR режимі (shadow_remove.py)
**Проблема:** Після `_remove_shadow_bgr` залишаються пікселі з кольоровим шумом.

**Файл:** `processing/shadow_remove.py`

**Кроки:**
- [ ] Після merge конвертувати у LAB
- [ ] Визначити колір фону з яскравих пікселів
- [ ] Маска артефактів: темний L або висока chroma
- [ ] Розширити маску на 3px, замінити на колір фону
- [ ] Відмітити виконання.

---

### 4.3 — `flat_background` поглинає скани (doc_classifier.py)
**Проблема:** Скан порожнього або майже порожнього аркуша класифікується як `flat_background` і пропускає обробку.

**Файл:** `processing/doc_classifier.py` → `_is_flat_background`

**Кроки:**
- [ ] Додати перевірку країв (edge detection)
- [ ] Збільшити `FLAT_BG_UNIFORMITY_THRESH` з 0.70 до 0.82
- [ ] Додати HoughLinesP перевірку
- [ ] Для `flat_background` застосовувати мінімальний pipeline
- [ ] Відмітити виконання.

---

### 4.4 — Ширина черги файлів регульована (main_window.py)
**Проблема:** `setFixedWidth` не дає змінювати ширину мишею.

**Файл:** `gui/main_window.py`

**Кроки:**
- [ ] Замінити ліву колонку і центр з `QHBoxLayout` на `QSplitter(Qt.Horizontal)`
- [ ] Ліву панель (черга) помістити у `QWidget` з min/max шириною
- [ ] Центральну панель (прев'ю) — у `QWidget` з `setMinimumWidth(600)`
- [ ] Зберігати/відновлювати `splitter.sizes()` у `_save_window_geometry` / `_load_window_geometry`
- [ ] Видалити `self._queue.setFixedWidth(queue_width)`
- [ ] Відмітити виконання.

---

### 4.5 — `mark_done` помилково позначає `▶` поточний файл (queue_view.py)
**Проблема:** В `_on_auto_done` перевірка `not startswith(("✓","✗"))` дозволяє позначити `▶ current` як виконаний.

**Файл:** `gui/queue_view.py` і `gui/main_window.py` → `_on_auto_done`

**Кроки:**
- [ ] У `_set_status` зберігати статус у `item.setData(UserRole+1, status)` — окремо від тексту
- [ ] У `_on_auto_done` перевіряти статус через `item.data(UserRole+1)` замість перевірки тексту
- [ ] Відмітити виконання.

---

### 4.6 — `_do_persp_reset` губить autofix (main_window.py)
**Проблема:** Якщо послідовність: autofix → ручна перспектива → "Скинути перспективу" — втрачається autofix.

**Файл:** `gui/main_window.py` → `_do_persp_reset`

**Кроки:**
- [ ] Додати поле `self._base_before_perspective` — знімок `_base` ДО будь-якої перспективи
- [ ] Зберігати при першому вході в perspective-режим
- [ ] У `_do_persp_reset`: повертатись до `_base_for_perspective` (зберігає autofix)
- [ ] Якщо `_base_for_perspective is None` → повертатись до `_orig`
- [ ] Очищати у `_clear_queue` і `_on_queue_selection`
- [ ] Відмітити виконання.

---

### 4.7 — Надійний друк замість `mspaint /pt` (core/printer.py)
**Проблема:** `mspaint /pt` — legacy, не підтримує параметри друку, ненадійний на Windows 11.

**Файл:** `core/printer.py`

**Кроки:**
- [ ] Додати спробу використати `win32print` (з `pywin32`) як основний метод
- [ ] Реалізувати `_print_windows_win32(path, printer_name)`
- [ ] Fallback ланцюжок: win32print → PowerShell Start-Process → mspaint /pt
- [ ] Додати `pywin32` до `requirements.txt` як опціональну залежність
- [ ] Відмітити виконання.

---

### 4.8 — Артефакти від `L_MIN_CLAMP` у shadow_remove (shadow_remove.py)
**Проблема:** Пікселі L=10-30 після ділення на малий background (~15-20) дають L=255 → білі плями там де має бути темний текст.

**Файл:** `processing/shadow_remove.py`

**Кроки:**
- [ ] Після `l_norm = cv2.divide(...)` застосувати маску захисту тексту
- [ ] Обмежити: нормалізований текст не може бути яскравішим за (оригінал * 2.5)
- [ ] Збільшити `L_MIN_CLAMP` з 5 до 15
- [ ] Перевірити на документі з чорним текстом
- [ ] Відмітити виконання.

---

### 4.9 — Перейменування "BGR" в UI (settings_window.py)
**Файл:** `gui/settings_window.py`

**Кроки:**
- [x] Рядок `"BGR-алгоритм (краще для деяких документів):"` → `"Канальний режим тіней (для складних випадків):"`
- [x] Оновлено tooltip
- [x] Відмітити виконання.

---

## БЛОК 5: Завершальні перевірки

### 5.1 — Регресійне тестування
- [ ] Відкрити файл 007.jpg → авто-перспектива → документ НЕ має відзеркалюватись
- [ ] Відкрити файл 008.jpg → авто-перспектива → документ НЕ має відзеркалюватись
- [ ] Швидко перемикати файли в черзі → прев'ю завжди показує поточний файл
- [ ] Скинути слайдери → різкість = 0.4, решта = 0.0
- [ ] Зміна режиму тіней → перезапуск → режим збережено
- [ ] Рух слайдера → UI не підвисає навіть на 4000×3000
- [ ] Зміна розміру вікна → в INI записується раз після зупинки (не під час руху)
- [ ] Закриття вікна під час пакетної обробки → не зависає більше ніж 0.5с
- [ ] Відмітити виконання.

---

## Порядок виконання (рекомендований)

| Черга | Блок | Статус |
|-------|------|--------|
| 1 | 1.1, 1.2, 1.4 | ✅ Виконано |
| 2 | 3.1, 3.2, 3.3 | ✅ Виконано |
| 3 | 3.4, 3.5, 3.6 | ✅ Виконано |
| 4 | 1.3, 1.5, 1.6 | ✅ Виконано |
| 5 | 2.1, 2.2 | ✅ Виконано |
| 6 | 4.1, 4.2 | ⬜ Залишилось |
| 7 | 4.3–4.9 | ⬜ Залишилось (крім 4.9) |
| 8 | 2.3–2.6 | ✅ Виконано |
| 9 | 5.1 | ⬜ Залишилось |