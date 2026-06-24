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

- [x] Відмітити виконання в TASK.md

---

## Крок 2 — Переписати BatchProcessor.run_auto() на паралельну обробку

**Файл:** `batch/batch_processor.py`

- [x] Відмітити виконання в TASK.md

---

## Крок 3 — Додати налаштування потоків у settings_window.py

**Файл:** `gui/settings_window.py`

- [x] Відмітити виконання в TASK.md

---

## Крок 4 — Оновити AutoWorker у main_window.py

**Файл:** `gui/main_window.py`

- [x] Відмітити виконання в TASK.md

---

## Крок 5 — Додати memory guard у BatchProcessor

**Файл:** `batch/batch_processor.py`

- [x] Відмітити виконання в TASK.md

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

- [x] Відмітити виконання в TASK.md