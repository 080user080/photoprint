"""
Тест перемикача Ч/Б: завантаження → ввімкнення Ч/Б → скріншот → вимикання → скріншот.
Для роботи потрібен реальний desktop з GUI (pytest.mark.gui).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester

SCENARIO_NAME = "test_grayscale_toggle"


def _run_scenario(tester: GUITester) -> bool:
    """Сценарій: завантаження → ввімкнення Ч/Б → скріншот."""
    print("=" * 60)
    print(f"Сценарій: {SCENARIO_NAME}")
    print("=" * 60)

    # 1. Запуск
    if not tester.launch_app():
        return False
    tester.wait(5)

    # 2. Активація вікна
    if not tester.activate_window(title="PhotoPrint", max_retries=10, retry_delay=1.0):
        print("[WARN] Вікно не знайдено")
    tester.wait(1)

    # 3. Пошук тестового зображення
    test_files = sorted(tester.test_images_dir.glob("*.jpg"))
    if not test_files:
        print("[ERROR] Немає тестових зображень")
        tester.close_app()
        return False
    test_image = test_files[0]
    abs_path = str(test_image.resolve())

    # 4. Drag & Drop
    print(f"[DROP] Завантаження: {abs_path}")
    tester.drag_drop_files([abs_path])
    tester.wait(3)

    # 5. Ввімкнення Ч/Б
    print("[ACTION] Вмикаємо Чорно-білий...")
    result = tester.click_widget("chk_grayscale")
    if not result:
        print("[WARN] Клік по objectName не вдався, пробуємо set_checkbox...")
        tester.set_checkbox("чорно-білий", checked=True)
    tester.wait(1)

    # 6. Скріншот в Ч/Б
    screenshot_bw = tester.results_dir / "test_grayscale_on.png"
    print(f"[SCREENSHOT] {screenshot_bw}")
    tester.screenshot_window(str(screenshot_bw))

    # 7. Вимикання Ч/Б
    print("[ACTION] Вимикаємо Чорно-білий...")
    tester.click_widget("chk_grayscale")
    tester.wait(1)

    # 8. Скріншот без Ч/Б
    screenshot_color = tester.results_dir / "test_grayscale_off.png"
    print(f"[SCREENSHOT] {screenshot_color}")
    tester.screenshot_window(str(screenshot_color))

    tester.close_app()
    print(f"\n{'=' * 60}")
    print(f"Сценарій {SCENARIO_NAME} ЗАВЕРШЕНО")
    print(f"{'=' * 60}")
    return True


def test_grayscale_toggle():
    """pytest entry point."""
    tester = GUITester("main.py", debug_mode=False)
    tester.setup_directories()
    tester.clear_logs()
    return _run_scenario(tester)


if __name__ == "__main__":
    success = test_grayscale_toggle()
    sys.exit(0 if success else 1)