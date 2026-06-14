"""
Тест ручної перспективи: завантаження → ручна точка → застосування → скріншот.
Для роботи потрібен реальний desktop з GUI (pytest.mark.gui).
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester

SCENARIO_NAME = "test_perspective_manual"


@pytest.mark.gui
def _run_scenario(tester: GUITester) -> bool:
    """Сценарій: завантаження → ручна перспектива → скріншот."""
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

    # 5. Натискаємо "Ручна перспектива" (через objectName)
    print("[ACTION] Натискаємо Ручна перспектива...")
    result = tester.click_widget("btn_persp_manual")
    if not result:
        print("[WARN] Клік по objectName не вдався, пробуємо click_button_by_text...")
        tester.click_button_by_text("ручна перспектива")
    tester.wait(3)

    # 6. Скріншот після ручної перспективи
    screenshot_path = tester.results_dir / "test_perspective_manual.png"
    print(f"[SCREENSHOT] {screenshot_path}")
    tester.screenshot_window(str(screenshot_path))

    # 7. Скидаємо перспективу
    print("[ACTION] Скидаємо перспективу...")
    tester.click_widget("btn_persp_reset")
    tester.wait(1)

    tester.close_app()
    print(f"\n{'=' * 60}")
    print(f"Сценарій {SCENARIO_NAME} ЗАВЕРШЕНО")
    print(f"{'=' * 60}")
    return True


@pytest.mark.gui
def test_perspective_manual():
    """pytest entry point."""
    tester = GUITester("main.py", debug_mode=False)
    tester.setup_directories()
    tester.clear_logs()
    return _run_scenario(tester)


if __name__ == "__main__":
    success = test_perspective_manual()
    sys.exit(0 if success else 1)