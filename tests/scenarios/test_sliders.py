"""Тест слайдерів: зміна всіх слайдерів та перевірка прев'ю"""
import os
import sys
from pathlib import Path

# Додаємо шлях до кореневої директорії проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def test_sliders():
    """Тест: завантаження -> зміна кожного слайдера -> скріншоти -> скидання"""
    print("=" * 60)
    print("Тест: слайдери (тіні, яскравість, контраст, різкість, HDR)")
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

    # 4. Drag & Drop тестового зображення
    test_image = tester.test_images_dir / "0 (3).jpg"
    if not test_image.exists():
        test_image = sorted(tester.test_images_dir.glob("*.jpg"))[0]
    abs_path = str(test_image.resolve())
    print(f"[DROP] Завантаження файлу: {abs_path}")
    if not tester.drag_drop_files([abs_path]):
        print("[WARN] Drag & Drop не спрацював")
    tester.wait(2)

    # 5. Скріншот оригіналу
    screenshot_orig = tester.results_dir / "01_original.png"
    print(f"[SCREENSHOT] Оригінал: {screenshot_orig}")
    tester.screenshot_window(str(screenshot_orig))
    tester.wait(0.5)

    # 6. Зміна тіней
    print("[ACTION] Зміна слайдера тіней (0.8)...")
    tester.drag_slider("тіні", 0.8)
    tester.wait(1)
    screenshot_shadows = tester.results_dir / "02_shadows.png"
    tester.screenshot_window(str(screenshot_shadows))
    tester.wait(0.5)

    # 7. Зміна яскравості
    print("[ACTION] Зміна слайдера яскравості (0.6)...")
    tester.drag_slider("яскравість", 0.6)
    tester.wait(1)
    screenshot_bright = tester.results_dir / "03_brightness.png"
    tester.screenshot_window(str(screenshot_bright))
    tester.wait(0.5)

    # 8. Зміна контрасту
    print("[ACTION] Зміна слайдера контрасту (0.7)...")
    tester.drag_slider("контраст", 0.7)
    tester.wait(1)
    screenshot_contrast = tester.results_dir / "04_contrast.png"
    tester.screenshot_window(str(screenshot_contrast))
    tester.wait(0.5)

    # 9. Зміна різкості
    print("[ACTION] Зміна слайдера різкості (0.5)...")
    tester.drag_slider("різкість", 0.5)
    tester.wait(1)
    screenshot_sharp = tester.results_dir / "05_sharpen.png"
    tester.screenshot_window(str(screenshot_sharp))
    tester.wait(0.5)

    # 10. Зміна HDR
    print("[ACTION] Зміна слайдера HDR (0.4)...")
    tester.drag_slider("hdr", 0.4)
    tester.wait(1)
    screenshot_hdr = tester.results_dir / "06_hdr.png"
    tester.screenshot_window(str(screenshot_hdr))
    tester.wait(0.5)

    # 11. Скидання всіх слайдерів
    print("[ACTION] Скидання слайдерів...")
    if not tester.click_button_by_text("Скинути слайдери"):
        print("[WARN] Кнопка 'Скинути слайдери' не знайдена")
    tester.wait(1)

    # 12. Скріншот після скидання
    screenshot_reset = tester.results_dir / "07_reset.png"
    print(f"[SCREENSHOT] Після скидання: {screenshot_reset}")
    tester.screenshot_window(str(screenshot_reset))
    tester.wait(0.5)

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
    success = test_sliders()
    sys.exit(0 if success else 1)
