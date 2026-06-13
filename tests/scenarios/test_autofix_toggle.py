"""Тест перемикання автофікс та слайдерів"""
import os
import sys
from pathlib import Path

# Додаємо шлях до кореневої директорії проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def test_autofix_toggle():
    """Тест: завантаження файлу -> Auto Fix -> скріншот -> зміна слайдерів -> скріншот"""
    print("=" * 60)
    print("Тест: Auto Fix + слайдери")
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

    # 4. Дебаг: перелічуємо дочірні віджети
    print("[DEBUG] Перелік віджетів вікна...")
    tester.list_child_widgets()
    tester.wait(0.5)

    # 5. Drag & Drop тестового зображення
    test_image = tester.test_images_dir / "0 (1).jpg"
    if test_image.exists():
        abs_path = str(test_image.resolve())
        print(f"[DROP] Завантаження файлу: {abs_path}")
        if not tester.drag_drop_files([abs_path]):
            print("[WARN] Drag & Drop не спрацював, спробуємо кнопку 'Додати файли'")
            tester.click_button_by_text("Додати файли")
            tester.wait(1)
            # Ввід шляху через діалогове вікно
            tester.type_text(abs_path)
            tester.press_key("enter")
    else:
        print(f"[ERROR] Тестовий файл не знайдено: {test_image}")
        tester.close_app()
        return False
    tester.wait(2)

    # 6. Скріншот ДО Auto Fix
    screenshot_before = tester.results_dir / "01_before_autofix.png"
    print(f"[SCREENSHOT] Скріншот ДО Auto Fix: {screenshot_before}")
    tester.screenshot_window(str(screenshot_before))
    tester.wait(0.5)

    # 7. Натискаємо кнопку "Auto Fix"
    print("[ACTION] Натискаємо Auto Fix...")
    if not tester.click_button_by_text("Auto Fix"):
        print("[WARN] Кнопка 'Auto Fix' не знайдена через API, пробуємо Alt+A")
        tester.hotkey("alt", "a")
    tester.wait(2)

    # 8. Скріншот ПІСЛЯ Auto Fix
    screenshot_after = tester.results_dir / "02_after_autofix.png"
    print(f"[SCREENSHOT] Скріншот ПІСЛЯ Auto Fix: {screenshot_after}")
    tester.screenshot_window(str(screenshot_after))
    tester.wait(0.5)

    # 9. Зміна слайдера яскравості
    print("[ACTION] Зміна слайдера яскравості...")
    tester.drag_slider("яскравість", 0.7)
    tester.wait(1)

    # 10. Скріншот ПІСЛЯ зміни яскравості
    screenshot_brightness = tester.results_dir / "03_after_brightness.png"
    print(f"[SCREENSHOT] Скріншот ПІСЛЯ яскравості: {screenshot_brightness}")
    tester.screenshot_window(str(screenshot_brightness))
    tester.wait(0.5)

    # 11. Зміна слайдера контрасту
    print("[ACTION] Зміна слайдера контрасту...")
    tester.drag_slider("контраст", 0.6)
    tester.wait(1)

    # 12. Скріншот ПІСЛЯ зміни контрасту
    screenshot_contrast = tester.results_dir / "04_after_contrast.png"
    print(f"[SCREENSHOT] Скріншот ПІСЛЯ контрасту: {screenshot_contrast}")
    tester.screenshot_window(str(screenshot_contrast))
    tester.wait(0.5)

    # 13. Перемикання Ч/Б
    print("[ACTION] Перемикання Ч/Б...")
    tester.set_checkbox("Чорно-білий", checked=True)
    tester.wait(1)

    # 14. Скріншот ПІСЛЯ Ч/Б
    screenshot_bw = tester.results_dir / "05_after_bw.png"
    print(f"[SCREENSHOT] Скріншот ПІСЛЯ Ч/Б: {screenshot_bw}")
    tester.screenshot_window(str(screenshot_bw))
    tester.wait(0.5)

    # 15. Скидання слайдерів
    print("[ACTION] Скидання слайдерів...")
    if not tester.click_button_by_text("Скинути слайдери"):
        print("[WARN] Кнопка 'Скинути слайдери' не знайдена")
    tester.wait(1)

    # 16. Скріншот ПІСЛЯ скидання
    screenshot_reset = tester.results_dir / "06_after_reset.png"
    print(f"[SCREENSHOT] Скріншот ПІСЛЯ скидання: {screenshot_reset}")
    tester.screenshot_window(str(screenshot_reset))
    tester.wait(0.5)

    # 17. Закриття додатка
    tester.close_app()

    # 18. Результати
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Скріншоти збережено в: {tester.results_dir}")
    for f in sorted(tester.results_dir.glob("*.png")):
        print(f"  {f.name}")

    return True


if __name__ == "__main__":
    success = test_autofix_toggle()
    sys.exit(0 if success else 1)
