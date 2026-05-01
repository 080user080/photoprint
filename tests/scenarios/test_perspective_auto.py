"""Тест авто-перспективи"""
import sys
from pathlib import Path

# Додаємо шлях до кореневої директорії проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def test_perspective_auto():
    """Тест авто-перспективи"""
    print("=" * 60)
    print("Тест: авто-перспектива")
    print("=" * 60)

    # Шлях до скрипта запуску GUI
    app_path = "gui/main.py"
    venv_python = None  # Використовувати поточний python

    # Створення тестувальника
    tester = GUITester(app_path, venv_python)

    # 1. Налаштування
    tester.setup_directories()
    tester.clear_logs()

    # 2. Запуск додатка
    print("🚀 Запуск PhotoPrint...")
    if not tester.launch_app():
        return False
    tester.wait(3)

    # 3. Активація вікна
    print("🔓 Активація вікна PhotoPrint...")
    if not tester.activate_window(title="PhotoPrint"):
        tester.close_app()
        return False
    tester.wait(1)

    # 4. Завантаження тестового зображення з перспективою
    # TODO: реалізувати завантаження через клавіатуру/мишку
    print("📁 Завантаження тестового зображення з перспективою...")
    print("⚠️  Не реалізовано - пропускаємо")
    tester.wait(1)

    # 5. Клік на кнопку авто-перспектива
    # TODO: реалізувати клік через координати
    print("🔲 Клік на кнопку авто-перспектива...")
    print("⚠️  Не реалізовано - пропускаємо")
    tester.wait(2)

    # 6. Збереження скріншоту
    screenshot_path = tester.results_dir / "perspective_auto.png"
    print(f"📸 Збереження скріншоту: {screenshot_path}")
    if not tester.screenshot(str(screenshot_path)):
        print("⚠️  Скріншот не збережено")

    # 7. Порівняння з очікуваним результатом
    expected_path = tester.expected_dir / "perspective_expected.jpg"
    if expected_path.exists():
        print(f"🔍 Порівняння з очікуваним результатом: {expected_path}")
        tester.compare_images(str(screenshot_path), str(expected_path), tolerance=10)
    else:
        print(f"⚠️  Очікуваний результат не знайдено: {expected_path}")

    # 8. Закриття
    tester.close_app()

    # 9. Перевірка результатів
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕНО")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_perspective_auto()
    sys.exit(0 if success else 1)
