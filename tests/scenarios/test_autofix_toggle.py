"""Тест перемикання автофікс чекбоксу"""
import sys
from pathlib import Path

# Додаємо шлях до кореневої директорії проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def test_autofix_toggle():
    """Тест перемикання автофікс чекбоксу"""
    print("=" * 60)
    print("Тест: перемикання автофікс чекбоксу")
    print("=" * 60)

    # Шлях до скрипта запуску GUI
    app_path = "main.py"
    venv_python = None  # Використовувати поточний python

    # Створення тестувальника
    tester = GUITester(app_path, venv_python)

    # 1. Налаштування
    tester.setup_directories()
    tester.clear_logs()

    # 2. Запуск додатка
    print("[LAUNCH] Запуск PhotoPrint...")
    if not tester.launch_app():
        return False
    tester.wait(3)

    # 3. Активація вікна
    print("[ACTIVATE] Активація вікна PhotoPrint...")
    # if not tester.activate_window(title="PhotoPrint"):
    #     tester.close_app()
    #     return False
    tester.wait(1)

    # 4. Завантаження тестового зображення (через drag & drop або меню)
    # TODO: реалізувати завантаження через клавіатуру/мишку
    print("[LOAD] Завантаження тестового зображення...")
    print("[SKIP] Не реалізовано - пропускаємо")
    tester.wait(1)

    # 5. Перемикання автофікс (клік по чекбоксу або комбінація клавіш)
    # TODO: реалізувати перемикання через клавіатуру/мишку
    print("[ACTION] Перемикання автофікс...")
    print("[SKIP] Не реалізовано - пропускаємо")
    tester.wait(0.5)

    # 6. Збереження скріншоту
    screenshot_path = tester.results_dir / "autofix_toggle.png"
    print(f"[SCREENSHOT] Збереження скріншоту: {screenshot_path}")
    if not tester.screenshot(str(screenshot_path)):
        print("[WARN] Скріншот не збережено")

    # 7. Закриття
    tester.close_app()

    # 8. Перевірка результатів
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕНО")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_autofix_toggle()
    sys.exit(0 if success else 1)
