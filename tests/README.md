# Автоматизований тестувальник GUI для PhotoPrint

Цей каталог містить інфраструктуру для автоматизованого тестування GUI PhotoPrint через зовнішнє керування (клавіатура/мишка).

## Структура

```
tests/
├── gui_tester.py          # Ядро тестувальника (GUITester клас)
├── requirements.txt       # Залежності для тестів
├── README.md             # Цей файл
├── scenarios/            # Сценарії тестів
│   ├── test_autofix_toggle.py
│   └── test_perspective_auto.py
├── test_images/          # Тестові зображення
├── expected/             # Очікувані результати
├── results/              # Результати тестів (скріншоти)
└── logs/                 # Логи додатка
```

## Встановлення

1. Встановіть залежності:
```bash
pip install -r tests/requirements.txt
```

## Використання

### Запуск окремого тесту

```bash
python tests/scenarios/test_autofix_toggle.py
python tests/scenarios/test_perspective_auto.py
```

### Створення нового сценарію

1. Створіть новий файл в `tests/scenarios/`
2. Імпортуйте GUITester:
```python
from tests.gui_tester import GUITester
```

3. Створіть функцію тесту:
```python
def test_my_feature():
    tester = GUITester("gui/main.py")
    tester.setup_directories()
    tester.launch_app()
    tester.wait(3)
    tester.activate_window(title="PhotoPrint")
    # Ваші дії...
    tester.close_app()
    return True
```

## GUITester API

### Методи

- `launch_app()` - Запускає GUI додаток через subprocess
- `activate_window(title)` - Активує вікно за назвою
- `type_text(text)` - Вставляє текст в активне вікно
- `press_key(key)` - Натискає клавішу
- `click_at(x, y)` - Клікає мишкою по координатах
- `wait(seconds)` - Чекає заданий час
- `screenshot(filename)` - Зберігає скріншот вікна
- `compare_images(actual, expected, tolerance)` - Порівнює зображення
- `close_app()` - Закриває GUI додаток
- `setup_directories()` - Створює необхідні директорії
- `clear_logs()` - Очищає логи перед тестом
- `read_logs()` - Читає логи після тесту

## TODO

- [ ] Реалізувати завантаження зображень через drag & drop
- [ ] Реалізувати перемикання елементів через координати
- [ ] Додати більше сценаріїв тестів
- [ ] Інтеграція з pytest
- [ ] CI/CD інтеграція

## Примітки

- Тестувальник використовує зовнішнє керування (pyautogui, pywin32)
- Не залежить від внутрішньої структури GUI
- Працює як реальний користувач
- Потрібно щоб GUI було відкрито і видимим під час тестів
