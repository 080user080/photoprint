"""Тест пакетної обробки: завантаження кількох файлів, черга, друк"""
import os
import sys
from pathlib import Path

# Додаємо шлях до кореневої директорії проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def test_batch_queue():
    """Тест: завантаження кількох файлів -> перевірка черги -> авто-режим -> скріншоти"""
    print("=" * 60)
    print("Тест: пакетна обробка черги")
    print("=" * 60)

    app_path = "main.py"
    tester = GUITester(app_path)

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

    # 4. Drag & Drop кількох файлів
    test_files = sorted(tester.test_images_dir.glob("*.jpg"))[:5]
    if not test_files:
        print("[ERROR] Немає тестових зображень")
        tester.close_app()
        return False

    abs_paths = [str(f.resolve()) for f in test_files]
    print(f"[DROP] Завантаження {len(abs_paths)} файлів...")
    if not tester.drag_drop_files(abs_paths):
        print("[WARN] Drag & Drop не спрацював")
    tester.wait(2)

    # 5. Скріншот черги
    screenshot_queue = tester.results_dir / "01_queue_loaded.png"
    print(f"[SCREENSHOT] Черга завантажена: {screenshot_queue}")
    tester.screenshot_window(str(screenshot_queue))
    tester.wait(0.5)

    # 6. Перевіряємо режим "Авто"
    print("[ACTION] Перевіряємо режим Авто...")
    # Режим Авто вже має бути встановлено за замовчуванням
    tester.wait(0.5)

    # 7. Натискаємо "Друкувати все"
    print("[ACTION] Натискаємо Друкувати все...")
    if not tester.click_button_by_text("Друкувати все"):
        print("[WARN] Кнопка 'Друкувати все' не знайдена")
    tester.wait(3)

    # 8. Скріншот після друку
    screenshot_print = tester.results_dir / "02_after_print_all.png"
    print(f"[SCREENSHOT] Після друку: {screenshot_print}")
    tester.screenshot_window(str(screenshot_print))
    tester.wait(0.5)

    # 9. Очищення черги
    print("[ACTION] Очищення черги...")
    if not tester.click_button_by_text("Очистити чергу"):
        print("[WARN] Кнопка 'Очистити чергу' не знайдена")
    tester.wait(1)

    # 10. Скріншот після очищення
    screenshot_clear = tester.results_dir / "03_queue_cleared.png"
    print(f"[SCREENSHOT] Черга очищена: {screenshot_clear}")
    tester.screenshot_window(str(screenshot_clear))
    tester.wait(0.5)

    # 11. Закриття
    tester.close_app()

    # 12. Результати
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Скріншоти збережено в: {tester.results_dir}")
    for f in sorted(tester.results_dir.glob("*.png")):
        print(f"  {f.name}")

    return True


if __name__ == "__main__":
    success = test_batch_queue()
    sys.exit(0 if success else 1)
