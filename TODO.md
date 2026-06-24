# TASK: Паралельна обробка файлів (ThreadPoolExecutor)

## Крок 1 — Додати налаштування кількості потоків у app_settings.py

**Файл:** `config/app_settings.py`

Додати константи:
```python
import os as _os

DEFAULT_WORKER_THREADS = min(_os.cpu_count() or 4, 8)
# Для Ryzen 7 7700: cpu_count()=16, обмежуємо 8 щоб не вичерпати RAM
# (кожен потік тримає в пам'яті 1 повноформатне зображення ~20-50MB)
DEFAULT_MAX_WORKER_THREADS = 16
DEFAULT_MIN_WORKER_THREADS = 1
```

У `load()` додати:
```python
"worker_threads": cfg.getint(
    "processing", "worker_threads",
    fallback=DEFAULT_WORKER_THREADS
),
```

У `save()` у секцію `cfg["processing"]` додати:
```python
cfg["processing"]["worker_threads"] = str(
    settings.get("worker_threads", DEFAULT_WORKER_THREADS)
)
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 2 — Переписати BatchProcessor.run_auto() на паралельну обробку

**Файл:** `batch/batch_processor.py`

**Додати імпорти на початку файлу:**
```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
```

**Додати константу:**
```python
DEFAULT_WORKER_THREADS = 4   # fallback якщо не передано через settings
PRINT_LOCK_TIMEOUT = 30.0    # секунд максимум чекати на принтер
```

**Замінити весь метод `run_auto()`** на нову реалізацію з двома фазами:

**Фаза 1 — паралельна обробка зображень** (ThreadPoolExecutor):
Кожен потік: `load → run_autofix → maybe_save` → повертає `(index, path, processed_image | Exception)`

**Фаза 2 — послідовний друк у правильному порядку**:
Зібрати результати відсортовані за індексом → print_image один за одним.

```python
def run_auto(
    self,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_error:    Callable[[int, str, str], None] | None = None,
) -> int:
    """
    Паралельна обробка файлів через ThreadPoolExecutor.
    Фаза 1: завантаження + pipeline — паралельно (N потоків).
    Фаза 2: друк — послідовно в порядку черги (щоб зберегти порядок).
    """
    s = self.settings
    n_workers = s.get("worker_threads", DEFAULT_WORKER_THREADS)
    printed = 0
    total = self.total

    # Лічильник для thread-safe прогресу
    progress_lock = threading.Lock()
    progress_counter = [0]   # список щоб мутабельний у closure

    def _process_one(idx: int, path: str):
        """
        Виконується у потоці ThreadPoolExecutor.
        Повертає (idx, path, processed_image) або кидає виняток.
        НЕ виконує друк — він залишається у головному потоці.
        """
        filename = os.path.basename(path)
        image = None
        processed = None
        try:
            image = loader.load(path)
            if s.get("autofix_enabled", True):
                processed, _, _ = pipeline.run_autofix(
                    image,
                    sharpen_strength=s.get("sharpen_strength", DEFAULT_SHARPEN_STRENGTH),
                    hdr_strength=s.get("hdr_strength", DEFAULT_HDR_STRENGTH),
                    use_hdr=s.get("hdr_in_autofix", True),
                    use_perspective=s.get("auto_perspective", True),
                    bw_binary=s.get("bw_binary", False),
                    classify_bw_std_thresh=s.get("classify_bw_std_thresh", DEFAULT_CLASSIFY_BW_STD_THRESH),
                    classify_edge_ratio_min=s.get("classify_edge_ratio_min", DEFAULT_CLASSIFY_EDGE_RATIO_MIN),
                    classify_line_count_min=s.get("classify_line_count_min", DEFAULT_CLASSIFY_LINE_COUNT_MIN),
                    shadow_highlight_strength=s.get("shadow_highlight_strength", DEFAULT_SHADOW_HIGHLIGHT_STRENGTH),
                    autofix_contrast=s.get("autofix_contrast", 0.15),
                    settings=s,
                )
            else:
                processed = image.copy()

            # Зберігаємо якщо потрібно
            self._maybe_save(processed, path)

            # Thread-safe оновлення прогресу
            with progress_lock:
                progress_counter[0] += 1
                cur = progress_counter[0]
            if on_progress:
                on_progress(cur, total, filename)

            return idx, path, processed

        finally:
            # Явне звільнення пам'яті після обробки
            if image is not None:
                del image
            # processed НЕ видаляємо — він потрібен для друку

    # --- Фаза 1: паралельна обробка ---
    # results_map: idx -> (path, processed_image | Exception)
    results_map: dict[int, tuple[str, object]] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx: dict[Future, int] = {
            executor.submit(_process_one, i, path): i
            for i, path in enumerate(self._files)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            path = self._files[idx]
            filename = os.path.basename(path)
            try:
                result_idx, result_path, processed = future.result()
                results_map[idx] = (result_path, processed)
            except Exception as exc:
                self._logger.error(
                    f"Помилка обробки файлу {filename}: {exc}", exc_info=True
                )
                if on_error:
                    on_error(idx, filename, str(exc))
                results_map[idx] = (path, exc)

    # --- Фаза 2: послідовний друк у порядку черги ---
    for i in range(total):
        if i not in results_map:
            continue
        path, result = results_map[i]
        filename = os.path.basename(path)

        if isinstance(result, Exception):
            continue   # вже повідомили через on_error

        processed = result
        try:
            printer_module.print_image(
                processed,
                printer_name=s.get("printer_name", ""),
                jpg_quality=s.get("jpg_quality", DEFAULT_JPG_QUALITY),
            )
            printed += 1
        except Exception as exc:
            self._logger.error(
                f"Помилка друку файлу {filename}: {exc}", exc_info=True
            )
            if on_error:
                on_error(i, filename, str(exc))
        finally:
            del processed
            results_map[i] = (path, None)   # звільняємо пам'ять

    self._index = total
    return printed
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 3 — Додати налаштування потоків у settings_window.py

**Файл:** `gui/settings_window.py`

У метод `_page_printer_mode()` у груп-бокс "Режим запуску" додати після `self._cb_minimize_to_tray`:

```python
import os as _os

self._spin_worker_threads = QSpinBox()
self._spin_worker_threads.setRange(1, _os.cpu_count() or 16)
self._spin_worker_threads.setToolTip(
    f"Кількість паралельних потоків обробки.\n"
    f"Ваш процесор: {_os.cpu_count()} логічних ядер.\n"
    f"Рекомендовано: {min(_os.cpu_count() or 4, 8)} "
    f"(більше = швидше, але більше RAM)"
)
_set_spinbox_minw(self._spin_worker_threads)
mode_form.addRow(
    f"Потоків обробки (1–{_os.cpu_count() or 16}):",
    self._spin_worker_threads
)
```

У `_apply_settings()` додати:
```python
from config.app_settings import DEFAULT_WORKER_THREADS
self._spin_worker_threads.setValue(
    s.get("worker_threads", DEFAULT_WORKER_THREADS)
)
```

У `_collect_settings()` додати:
```python
"worker_threads": self._spin_worker_threads.value(),
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 4 — Оновити AutoWorker у main_window.py

**Файл:** `gui/main_window.py`

**Проблема:** прогрес-бар зараз показує `cur/total` де `cur` — порядковий номер
завершеного файлу. При паралельній обробці файли завершуються не по черзі.

Сигнал `progress` вже емітується thread-safe з `progress_counter` у `_process_one`.
Але `_on_auto_progress()` викликає `self._queue.mark_current(cur - 1)` — це некоректно
при паралельній обробці (файл `cur-1` може ще оброблятися).

**Що зробити:** У `_on_auto_progress()` замінити `mark_current` на `mark_done`:

```python
def _on_auto_progress(self, cur: int, total: int, fname: str):
    self._progress.setValue(cur)
    # При паралельній обробці не знаємо точний індекс —
    # просто оновлюємо статус без позначення конкретного рядка
    self._set_status(f"Оброблено {cur}/{total}: {fname}")
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 5 — Додати memory guard у BatchProcessor

**Файл:** `batch/batch_processor.py`

**Проблема:** при 16 потоках і зображеннях 4000×3000 px (~34MB кожне) одночасно
в RAM буде 16 × 34MB = 544MB тільки для вхідних зображень + стільки ж для оброблених.

Додати константи:
```python
# Обмеження RAM: не більше ніж MAX_RAM_RATIO від доступної
RAM_GUARD_ENABLED = True
MAX_RAM_RATIO = 0.6   # використовуємо максимум 60% RAM
BYTES_PER_WORKER_ESTIMATE = 150 * 1024 * 1024  # 150MB на потік (вхід + вихід + pipeline)
```

Додати функцію `_safe_worker_count()` у `BatchProcessor`:
```python
def _safe_worker_count(self, requested: int) -> int:
    """
    Зменшує кількість потоків якщо доступної RAM недостатньо.
    Якщо psutil не встановлено — повертає requested без змін.
    """
    if not RAM_GUARD_ENABLED:
        return requested
    try:
        import psutil
        available = psutil.virtual_memory().available
        max_by_ram = max(1, int(available * MAX_RAM_RATIO / BYTES_PER_WORKER_ESTIMATE))
        safe = min(requested, max_by_ram)
        if safe < requested:
            self._logger.info(
                f"RAM guard: знижено потоки {requested}→{safe} "
                f"(доступно {available // 1024 // 1024}MB)"
            )
        return safe
    except ImportError:
        return requested
```

У `run_auto()` замінити рядок:
```python
n_workers = s.get("worker_threads", DEFAULT_WORKER_THREADS)
```
На:
```python
n_workers = self._safe_worker_count(
    s.get("worker_threads", DEFAULT_WORKER_THREADS)
)
```

- [ ] Відмітити виконання в TASK.md

---

## Крок 6 — Перевірка та тест продуктивності

Перевірити що:
1. Програма запускається без помилок
2. У Диспетчері завдань при запуску "Друкувати все" з кількома файлами
   видно завантаження **кількох ядер** (не одного)
3. Налаштування "Потоків обробки" зберігається і відновлюється
4. Порядок друку відповідає порядку файлів у черзі
5. При помилці одного файлу інші продовжують оброблятися

Опціонально — встановити `psutil` для memory guard:
```bash
pip install psutil
```

- [ ] Відмітити виконання в TASK.md