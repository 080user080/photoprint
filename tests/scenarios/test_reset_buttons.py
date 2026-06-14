"""
Тест кнопок скидання: завантаження → Auto Fix → скидання перспективи → скидання всього → скріншоти.
Для роботи потрібен реальний desktop з GUI (pytest.mark.gui).
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester

SCENARIO_NAME = "test_reset_buttons"


@pytest.mark.gui
def _run_scenario(tester: GUITester) -> bool:
    """Сценарій: Auto Fix → Скинути перспективу → Скинути слайдери → скріншот."""
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

    # 5. Auto Fix
    print("[ACTION] Натискаємо Auto Fix...")
    result = tester.click_widget("btn_autofix")
    if not result:
        tester.click_button_by_text("auto fix")
    tester.wait(2)

    # 6. Скріншот після Auto Fix
    screenshot_autofix = tester.results_dir / "test_reset_after_autofix.png"
    print(f"[SCREENSHOT] {screenshot_autofix}")
    tester.screenshot_window(str(screenshot_autofix))

    # 7. Скидаємо перспективу
    print("[ACTION] Скидаємо перспективу...")
    result = tester.click_widget("btn_persp_reset")
    if not result:
        tester.click_button_by_text("скинути перспективу")
    tester.wait(1)

    # 8. Скріншот після скидання перспективи
    screenshot_persp_reset = tester.results_dir / "test_reset_persp.png"
    print(f"[SCREENSHOT] {screenshot_persp_reset}")
    tester.screenshot_window(str(screenshot_persp_reset))

    # 9. Скидаємо слайдери
    print("[ACTION] Скидаємо слайдери...")
    result = tester.click_widget("btn_reset_sliders")
    if not result:
        tester.click_button_by_text("скинути слайдери")
    tester.wait(1)

    # 10. Скріншот після скидання слайдерів
    screenshot_sliders_reset = tester.results_dir / "test_reset_sliders.png"
    print(f"[SCREENSHOT] {screenshot_sliders_reset}")
    tester.screenshot_window(str(screenshot_sliders_reset))

    tester.close_app()
    print(f"\n{'=' * 60}")
    print(f"Сценарій {SCENARIO_NAME} ЗАВЕРШЕНО")
    print(f"{'=' * 60}")
    return True


@pytest.mark.gui
def test_reset_buttons():
    """pytest entry point."""
    tester = GUITester("main.py", debug_mode=False)
    tester.setup_directories()
    tester.clear_logs()
    return _run_scenario(tester)


if __name__ == "__main__":
    success = test_reset_buttons()
    sys.exit(0 if success else 1)