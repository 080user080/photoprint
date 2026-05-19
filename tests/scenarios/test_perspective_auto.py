"""Тест авто-перспективи та скидання"""
import os
import sys
from pathlib import Path

# Додаємо шлях до кореневої директорії проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def test_perspective_auto():
    """Тест: завантаження -> Auto Fix -> авто-перспектива -> скріншот -> скидання перспективи"""
    print("=" * 60)
    print("Тест: авто-перспектива")
    print("=" * 60)

    # Шлях до скрипта запуску GUI
    app_path = "main.py"
    venv_python = None

    # Створення тестувальника
    tester = GUITester(app_path, venv_python)

    # 1. Налаштування
    tester.setup_directories()
    tester.clear_logs()

    # 2. Запуск додатка
    print("[LAUNCH] Запуск PhotoPrint...")
    if not tester.launch_app():
        return False
    tester.wait(5)

    # 3. Активація вікна
    print("[ACTIVATE] Активація вікна PhotoPrint...")
    if not tester.activate_window(title="PhotoPrint", max_retries=10, retry_delay=1.0):
        print("[WARN] Вікно не знайдено, продовжуємо...")
    tester.wait(1)

    # 4. Drag & Drop тестового зображення (документ з перспективою)
    # Використовуємо файли з тестової папки
    test_files = sorted(tester.test_images_dir.glob("*.jpg"))
    if not test_files:
        print("[ERROR] Немає тестових зображень")
        tester.close_app()
        return False

    # Беремо перший файл
    test_image = test_files[0]
    abs_path = str(test_image.resolve())
    print(f"[DROP] Завантаження файлу: {abs_path}")
    if not tester.drag_drop_files([abs_path]):
        print("[WARN] Drag & Drop не спрацював")
    tester.wait(2)

    # 5. Скріншот ДО корекції
    screenshot_before = tester.results_dir / "01_before_persp.png"
    print(f"[SCREENSHOT] ДО перспективи: {screenshot_before}")
    tester.screenshot_window(str(screenshot_before))
    tester.wait(0.5)

    # 6. Натискаємо Auto Fix (авто-перспектива працює після автофікс)
    print("[ACTION] Натискаємо Auto Fix...")
    if not tester.click_button_by_text("Auto Fix"):
        print("[WARN] Кнопка 'Auto Fix' не знайдена")
    tester.wait(2)

    # 7. Скріншот ПІСЛЯ Auto Fix
    screenshot_autofix = tester.results_dir / "02_after_autofix.png"
    print(f"[SCREENSHOT] ПІСЛЯ Auto Fix: {screenshot_autofix}")
    tester.screenshot_window(str(screenshot_autofix))
    tester.wait(0.5)

    # 8. Натискаємо "Авто-перспектива"
    print("[ACTION] Натискаємо Авто-перспектива...")
    if not tester.click_button_by_text("Авто-перспектива"):
        print("[WARN] Кнопка 'Авто-перспектива' не знайдена через API")
    tester.wait(3)

    # 9. Скріншот ПІСЛЯ авто-перспективи
    screenshot_persp = tester.results_dir / "03_after_perspective.png"
    print(f"[SCREENSHOT] ПІСЛЯ авто-перспективи: {screenshot_persp}")
    tester.screenshot_window(str(screenshot_persp))
    tester.wait(0.5)

    # 10. Натискаємо "Скинути перспективу"
    print("[ACTION] Скидаємо перспективу...")
    if not tester.click_button_by_text("Скинути перспективу"):
        print("[WARN] Кнопка 'Скинути перспективу' не знайдена")
    tester.wait(1)

    # 11. Скріншот ПІСЛЯ скидання перспективи
    screenshot_reset = tester.results_dir / "04_after_persp_reset.png"
    print(f"[SCREENSHOT] ПІСЛЯ скидання перспективи: {screenshot_reset}")
    tester.screenshot_window(str(screenshot_reset))
    tester.wait(0.5)

    # 12. Порівняння з очікуваним результатом (якщо є)
    expected_path = tester.expected_dir / "perspective_expected.jpg"
    if expected_path.exists():
        print(f"[COMPARE] Порівняння з очікуваним: {expected_path}")
        tester.compare_images(str(screenshot_persp), str(expected_path), tolerance=10)
    else:
        print(f"[INFO] Очікуваний результат не знайдено: {expected_path}")

    # 13. Закриття
    tester.close_app()

    # 14. Результати
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Скріншоти збережено в: {tester.results_dir}")
    for f in sorted(tester.results_dir.glob("*.png")):
        print(f"  {f.name}")

    return True


if __name__ == "__main__":
    success = test_perspective_auto()
    sys.exit(0 if success else 1)
