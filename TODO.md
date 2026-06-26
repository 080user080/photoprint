## Завдання 1 — Виправити модель даних зображення

**Файл:** `gui/main_window.py`

**Проблема детально:**
- `_base` оновлюється після авто-яскравості/контрасту, але **не** після `_do_autofix_classic()` — там результат іде тільки в `_processed`
- Слайдери в `_on_controls_changed` завжди беруть `_base` — тому якщо `_base` не містить автофіксу, все відкочується
- Ручна перспектива: `_on_persp_pts` пише в `_processed`, але `_base` залишається старим — після будь-якого руху слайдера перспектива зникає

**Поточний проблемний код `_do_autofix_classic`:**
```python
# ЗАРАЗ — результат НЕ іде в _base
self._processed = result
self._preview.set_after(image_utils.make_preview(result))
```

**Поточний проблемний код `_on_controls_changed`:**
```python
# ЗАРАЗ — завжди бере _base, який не містить автофікс
result = pipeline.run_manual_adjustments(
    self._base,   # <-- _base не містить результату autofix!
    ...
)
```

**Що змінити:**

В `__init__` додати чіткий коментар до кожного поля:
```python
self._orig: Optional[np.ndarray] = None      # НЕЗМІННИЙ оригінал з диску
self._base: Optional[np.ndarray] = None      # після autofix + перспективи + авто-корекцій
self._processed: Optional[np.ndarray] = None # _base + поточні слайдери (фінал для друку)
```

В `_do_autofix_classic` — **після** отримання `result` записати його в `_base`:
```python
# БУЛО:
self._processed = result
self._preview.set_after(image_utils.make_preview(result))

# СТАЛО:
self._base = result.copy()   # <-- зберігаємо як нову базу
self._processed = result
self._preview.set_after(image_utils.make_preview(result))
```

В `_do_persp_auto` вже є `self._base = result.copy()` — перевірити що це стоїть **до** виклику `_on_controls_changed`.

В `_do_persp_manual` — ручна перспектива. Зараз `_on_persp_pts` пише тільки в `_processed`. Потрібно при **підтвердженні** (кнопка Друк або окрема кнопка "Застосувати перспективу") фіксувати результат у `_base`:
```python
def _do_print_current(self):
    # НА ПОЧАТКУ — фіксуємо ручну перспективу в _base якщо є
    if self._base_for_perspective is not None and self._processed is not None:
        self._base = self._processed.copy()   # вже є ця логіка, але перевірити
        self._base_for_perspective = None
    ...
```

Додати **окрему кнопку** `btn_apply_perspective = QPushButton("✔ Застосувати перспективу")` яка з'являється тільки коли активний режим ручної перспективи (`_base_for_perspective is not None`). При натисканні:
```python
def _do_apply_perspective(self):
    if self._processed is not None:
        self._base = self._processed.copy()
        self._base_for_perspective = None
        self._preview.disable_perspective_edit()
        self._btn_apply_perspective.setVisible(False)
        self._on_controls_changed()  # перерахувати _processed з нового _base + слайдери
```

В `_do_persp_manual` і `_do_persp_auto` — показувати цю кнопку:
```python
self._btn_apply_perspective.setVisible(True)
```

В `_do_persp_reset` — ховати кнопку:
```python
self._btn_apply_perspective.setVisible(False)
```

В `_do_reset_all` — повне скидання до `_orig`:
```python
def _do_reset_all(self):
    if self._orig is None:
        return
    self._base = self._orig.copy()
    self._base_for_perspective = None
    self._processed = self._orig.copy()
    self._perspective_corners = None
    self._preview.set_before(image_utils.make_preview(self._orig))
    self._preview.set_after(image_utils.make_preview(self._orig))
    self._preview.disable_perspective_edit()
    self._preview.set_autofix_applied(None)
    self._btn_apply_perspective.setVisible(False)
    self._set_status("Всі корекції скинуто до оригіналу")
    self._store_current_settings()
    self._update_buttons()
```

---

## Завдання 2 — Фоновий потік для Auto Fix та важких операцій

**Файл:** `gui/main_window.py`

**Проблема детально:** `_do_autofix_classic()`, `_do_persp_auto()`, `_do_auto_brightness()`, `_do_auto_contrast()`, `_do_auto_sharpen()` — всі викликаються синхронно в GUI-потоці. Pipeline з видаленням тіней займає 5–20 секунд. Вікно заморожується, ніякого відгуку.

**Що додати — новий клас `SingleImageWorker`** поруч з існуючим `AutoWorker`:

```python
class SingleImageWorker(QObject):
    """
    Виконує одну операцію обробки зображення у фоновому потоці.
    func: callable() -> np.ndarray  (або tuple)
    """
    finished = pyqtSignal(object)   # результат операції (np.ndarray або tuple)
    error    = pyqtSignal(str)      # повідомлення про помилку

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

**Додати в `__init__` MainWindow:**
```python
self._single_worker: Optional[SingleImageWorker] = None
self._single_thread: Optional[QThread] = None
```

**Додати хелпер `_run_in_background`:**
```python
def _run_in_background(self, func, on_finished, button_to_lock=None):
    """
    Запускає func() у фоновому потоці.
    on_finished(result) викликається в GUI-потоці після завершення.
    button_to_lock — кнопка яку заблокувати під час виконання.
    """
    if self._single_thread is not None and self._single_thread.isRunning():
        self._set_status("⏳ Зачекайте, попередня операція ще виконується")
        return

    # Показати прогрес
    self._progress.setRange(0, 0)   # indeterminate (пульсуючий)
    self._progress.setVisible(True)
    if button_to_lock:
        button_to_lock.setEnabled(False)
        button_to_lock.setText("⏳ Обробка…")
    self._set_buttons_enabled(False)

    self._single_thread = QThread()
    self._single_worker = SingleImageWorker(func)
    self._single_worker.moveToThread(self._single_thread)

    self._single_thread.started.connect(self._single_worker.run)

    def _on_done(result):
        self._progress.setVisible(False)
        self._set_buttons_enabled(True)
        if button_to_lock:
            button_to_lock.setEnabled(True)
            button_to_lock.setText("⚡ Auto Fix")   # або оригінальний текст
        self._single_thread.quit()
        self._single_thread = None
        self._single_worker = None
        on_finished(result)

    def _on_error(msg):
        self._progress.setVisible(False)
        self._set_buttons_enabled(True)
        if button_to_lock:
            button_to_lock.setEnabled(True)
            button_to_lock.setText("⚡ Auto Fix")
        self._single_thread.quit()
        self._single_thread = None
        self._single_worker = None
        self._set_status(f"Помилка: {msg}")
        self._logger.error(msg)

    self._single_worker.finished.connect(_on_done)
    self._single_worker.error.connect(_on_error)
    self._single_thread.start()
```

**Переробити `_do_autofix_classic`:**
```python
def _do_autofix_classic(self):
    if self._orig is None:
        self._set_status("Спочатку оберіть файл")
        return
    if self._single_thread is not None and self._single_thread.isRunning():
        self._set_status("⏳ Зачекайте, операція ще виконується")
        return

    s = self._settings
    vals = self._controls.values()
    base_snapshot = self._base.copy()   # знімок щоб не передавати self в інший потік

    def _work():
        if s.get("autofix_enabled", True):
            result, status_msg, log_entries = pipeline.run_autofix(
                base_snapshot,
                sharpen_strength=vals["sharpen_strength"],
                hdr_strength=vals["hdr_strength"],
                use_hdr=s.get("hdr_in_autofix", True),
                use_perspective=s.get("auto_perspective", False),
                partial_perspective=s.get("partial_perspective", False),
                bw_binary=s.get("bw_binary", False),
                classify_bw_std_thresh=s.get("classify_bw_std_thresh", 20.0),
                classify_edge_ratio_min=s.get("classify_edge_ratio_min", 0.03),
                classify_line_count_min=s.get("classify_line_count_min", 3),
                shadow_highlight_strength=vals["shadow_highlight"],
                output_color_mode=s.get("output_color_mode", "auto"),
                autofix_contrast=s.get("autofix_contrast", 0.15),
                contrast_mode=s.get("contrast_mode", "linear"),
                settings=s,
            )
            if s.get("autofix_enabled", True) and vals["grayscale"]:
                result = pipeline.run_grayscale(result)
            return result, status_msg, log_entries
        else:
            result = pipeline.run_manual_adjustments(
                base_snapshot,
                brightness=vals["brightness"],
                contrast=vals["contrast"],
                sharpen_strength=vals["sharpen_strength"],
                hdr_strength=vals["hdr_strength"],
                grayscale=vals["grayscale"],
                shadow_highlight_strength=vals["shadow_highlight"],
                contrast_mode=s.get("contrast_mode", "linear"),
            )
            return result, "Ручні налаштування", []

    def _on_done(payload):
        result, status_msg, log_entries = payload
        self._base = result.copy()      # <-- ФІКСУЄМО в _base
        self._processed = result
        self._preview.set_after(image_utils.make_preview(result))
        self._preview.set_autofix_applied("auto_fix")
        self._set_status(status_msg)
        self._update_buttons()

    self._run_in_background(_work, _on_done, button_to_lock=self._btn_autofix)
```

**Аналогічно переробити `_do_persp_auto`** — обгорнути виклик `pipeline.run_perspective_auto_smart` у лямбду і передати в `_run_in_background`. Callback `_on_done` має оновити `_base`, прев'ю, і показати точки.

**Аналогічно `_do_auto_brightness`, `_do_auto_contrast`, `_do_auto_sharpen`** — кожну обгорнути у `_run_in_background`.

---

## Завдання 3 — Прискорити shadow_remove через downscale

**Файл:** `processing/shadow_remove.py`

**Проблема детально:** `cv2.morphologyEx` з ядром 150–200px на зображенні 4000×3000px — це мільярди операцій. Модель фону принципово низькочастотна (тінь — великий плавний градієнт), тому точність на 4K не потрібна.

**Змінити функцію `_create_background_model`:**

```python
# Нові константи у верхній частині файлу
MORPH_MAX_ANALYSIS_SIDE = 1000   # макс. розмір для морфології (пікселів)

def _create_background_model(l_channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Створює модель фону. Для прискорення — зменшує зображення перед
    морфологією, потім апскейлить результат.
    """
    orig_h, orig_w = l_channel.shape[:2]
    max_side = max(orig_h, orig_w)

    # --- Downscale для прискорення ---
    if max_side > MORPH_MAX_ANALYSIS_SIDE:
        scale = MORPH_MAX_ANALYSIS_SIDE / max_side
        small_w = max(1, int(orig_w * scale))
        small_h = max(1, int(orig_h * scale))
        l_small = cv2.resize(l_channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
        # Масштабуємо kernel пропорційно, залишаємо непарним
        scaled_kernel = max(MORPH_KERNEL_MIN, int(kernel_size * scale) | 1)
    else:
        l_small = l_channel
        scaled_kernel = max(MORPH_KERNEL_MIN, kernel_size | 1)

    # --- Морфологія на зменшеному зображенні ---
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (scaled_kernel, scaled_kernel)
    )
    closed = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, morph_kernel)

    smooth_size = max(5, scaled_kernel // MORPH_SMOOTH_FACTOR) | 1
    background_small = cv2.GaussianBlur(closed, (smooth_size, smooth_size), BLUR_SIGMA)

    # --- Апскейл назад до оригінального розміру ---
    if max_side > MORPH_MAX_ANALYSIS_SIDE:
        background = cv2.resize(background_small, (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)
    else:
        background = background_small

    return background
```

**Додати в `main.py` після створення `QApplication`:**
```python
import cv2
# Дозволяємо OpenCV використовувати всі доступні ядра
cv2.setNumThreads(cv2.getNumberOfCPUs())
# Спробувати OpenCL якщо є сумісна відеокарта
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
```

---

## Завдання 4 — Замінити лог-віджет на статусний рядок

**Файл:** `gui/main_window.py`

**Що видалити:**
- `self._log_widget = QListWidget()` та всі посилання на нього
- `self._log_widget.setMaximumHeight(120)`
- `self._log_widget.setVisible(False)`
- метод `_show_log()`
- виклик `self._show_log(log_entries)` в `_do_autofix_classic`
- `center.addWidget(self._log_widget)` з layout

**Що додати — `QStatusBar`:**

В `_build_ui` наприкінці, після формування всього layout:
```python
# Статусний рядок вбудований у QMainWindow
sb = self.statusBar()
sb.setStyleSheet("QStatusBar { color: #444444; font-size: 12px; background: #F0F2F5; }")

# Постійний правий індикатор — поточний файл
self._status_file_label = QLabel("")
self._status_file_label.setStyleSheet("color: #888888; font-size: 11px; padding-right: 8px;")
sb.addPermanentWidget(self._status_file_label)
```

Замінити `self._status` (поточний QLabel) на виклики `self.statusBar().showMessage(text, timeout_ms)`:
```python
def _set_status(self, text: str, timeout_ms: int = 0):
    self.statusBar().showMessage(text, timeout_ms)

def _set_file_status(self, filename: str):
    self._status_file_label.setText(filename)
```

Замінити метод `_show_log` на скорочений вивід у статусбар:
```python
def _show_log(self, log_entries: list[dict]):
    """Виводить стислий підсумок у статусний рядок."""
    applied = [e["detail"] for e in log_entries if e.get("applied")]
    if applied:
        summary = "✓ " + " · ".join(applied[:4])   # перші 4 щоб не переповнювати
        if len(applied) > 4:
            summary += f" (+{len(applied)-4})"
        self.statusBar().showMessage(summary)
```

Старий `self._status = QLabel(...)` і `center.addWidget(self._status)` — видалити. Видалити також `self._status.setWordWrap(True)` і всі посилання.

---

## Завдання 5 — Переробити перемикач режиму видалення тіней

**Файл:** `gui/main_window.py`

**Що видалити:**
```python
# Видалити:
self._combo_shadow_mode = QComboBox()
self._combo_shadow_mode.addItem(...)
...
self._combo_shadow_mode.currentIndexChanged.connect(self._on_shadow_mode_changed)
buttons_row.addWidget(self._combo_shadow_mode)
```

**Що додати** — компактний блок з трьох радіокнопок. Розмістити в `buttons_row` між `btn_autofix` і `btn_print`:

```python
from PyQt6.QtWidgets import QButtonGroup

# Група радіокнопок для режиму тіней
shadow_group_widget = QWidget()
shadow_layout = QHBoxLayout(shadow_group_widget)
shadow_layout.setContentsMargins(4, 0, 4, 0)
shadow_layout.setSpacing(2)

shadow_label = QLabel("Тіні:")
shadow_label.setStyleSheet("color: #555; font-size: 11px;")
shadow_layout.addWidget(shadow_label)

self._rb_shadow_auto   = QRadioButton("авто")
self._rb_shadow_always = QRadioButton("завжди")
self._rb_shadow_never  = QRadioButton("ніколи")

for rb in (self._rb_shadow_auto, self._rb_shadow_always, self._rb_shadow_never):
    rb.setStyleSheet("color: #111; font-size: 12px;")
    shadow_layout.addWidget(rb)

self._rb_shadow_auto.setChecked(True)
self._rb_shadow_auto.setToolTip(
    "Алгоритм автоматично визначає чи є тіні на документі.\n"
    "Якщо тіней не знайдено — обробка пропускається."
)
self._rb_shadow_always.setToolTip(
    "Видаляти тіні завжди, навіть якщо алгоритм не виявив їх.\n"
    "Корисно для документів зі слабкими або нерівномірними тінями."
)
self._rb_shadow_never.setToolTip("Не обробляти тіні. Прискорює обробку.")

self._shadow_mode_group = QButtonGroup()
self._shadow_mode_group.addButton(self._rb_shadow_auto,   0)
self._shadow_mode_group.addButton(self._rb_shadow_always, 1)
self._shadow_mode_group.addButton(self._rb_shadow_never,  2)
self._shadow_mode_group.idClicked.connect(self._on_shadow_mode_changed)

buttons_row.addWidget(shadow_group_widget)
```

**Оновити `_on_shadow_mode_changed`:**
```python
def _on_shadow_mode_changed(self, btn_id: int):
    mode_map = {0: "auto", 1: "always", 2: "never"}
    mode = mode_map.get(btn_id, "auto")
    self._settings["shadow_remove_mode"] = mode
    app_settings.save(self._settings)
    self._do_autofix_classic()
```

**Оновити `_apply_default_mode`** — прибрати логіку ComboBox, замінити на:
```python
shadow_mode = self._settings.get("shadow_remove_mode", "auto")
mode_to_btn = {"auto": self._rb_shadow_auto, "always": self._rb_shadow_always, "never": self._rb_shadow_never}
mode_to_btn.get(shadow_mode, self._rb_shadow_auto).setChecked(True)
```

---

## Завдання 6 — Перейменувати режими та покращити їх опис

**Файл:** `gui/main_window.py`

**Що змінити в `_build_ui`:**
```python
# БУЛО:
self._radio_auto   = QRadioButton("Авто")
self._radio_manual = QRadioButton("Ручний")

# СТАЛО:
self._radio_auto   = QRadioButton("Пакетний")
self._radio_manual = QRadioButton("Покроковий")

self._radio_auto.setToolTip(
    "Пакетний режим: всі файли з черги будуть автоматично\n"
    "оброблені та надруковані без зупинок.\n"
    "Кнопка «Друкувати все» запускає повну чергу."
)
self._radio_manual.setToolTip(
    "Покроковий режим: кожне фото відкривається окремо.\n"
    "Можна відредагувати слайдерами, переглянути результат,\n"
    "потім надрукувати або пропустити."
)
```

**Додати оновлення статусу при перемиканні режиму.** Підписатися на сигнал групи кнопок:
```python
self._mode_group.idClicked.connect(self._on_mode_changed)
```

```python
def _on_mode_changed(self, btn_id: int):
    if btn_id == MODE_AUTO_ID:
        self._set_status(
            "Пакетний режим: натисніть «Друкувати все» щоб обробити всю чергу"
        )
    else:
        self._set_status(
            "Покроковий режим: редагуйте кожне фото та натискайте «Друк»"
        )
    self._update_buttons()
```

**Обмежити кнопку "Друкувати все" пакетним режимом в `_update_buttons`:**
```python
def _update_buttons(self):
    has_queue = bool(self._queue.get_all_paths())
    has_img   = self._orig is not None
    is_batch  = self._radio_auto.isChecked()

    self._btn_print_all.setEnabled(has_queue and is_batch)
    self._btn_print_all.setToolTip(
        "" if is_batch else "Доступно тільки в Пакетному режимі"
    )
    self._btn_print.setEnabled(has_img)
    self._btn_skip.setEnabled(self._processor.has_next() and not is_batch)
    self._btn_autofix.setEnabled(has_img)
    self._btn_save_img.setEnabled(has_img)
    
    # Кнопка застосування перспективи — видима тільки коли є незафіксована перспектива
    has_pending_persp = self._base_for_perspective is not None
    self._btn_apply_perspective.setVisible(has_pending_persp)
```