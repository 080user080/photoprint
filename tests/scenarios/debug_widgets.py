"""Дебаг-скрипт: перелічує всі віджети вікна PhotoPrint для визначення координат"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.gui_tester import GUITester


def debug_widgets():
    """Запускає PhotoPrint і перелічує всі віджети з координатами"""
    print("=" * 60)
    print("Дебаг: перелік віджетів PhotoPrint")
    print("=" * 60)

    app_path = "main.py"
    tester = GUITester(app_path)

    # Запуск додатка
    print("[LAUNCH] Запуск PhotoPrint...")
    if not tester.launch_app():
        return False
    tester.wait(5)

    # Активація вікна
    print("[ACTIVATE] Активація вікна PhotoPrint...")
    if not tester.activate_window(title="PhotoPrint", max_retries=10, retry_delay=1.0):
        print("[WARN] Вікно не знайдено, спробуємо перелічити всі вікна...")
        # Перелічити всі видимі вікна для дебагу
        try:
            import win32gui
            def enum_cb(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        results.append((hwnd, title))
                return True
            windows = []
            win32gui.EnumWindows(enum_cb, windows)
            print("[DEBUG] Видимі вікна:")
            for hwnd, title in windows:
                print(f"  hwnd={hwnd} title='{title}'")
        except Exception as e:
            print(f"[ERROR] {e}")
        tester.close_app()
        return False
    tester.wait(1)

    # Координати вікна
    rect = tester.find_window_rect()
    if rect:
        left, top, right, bottom = rect
        print(f"[INFO] Вікно: ({left},{top}) - ({right},{bottom}), розмір: {right-left}x{bottom-top}")

    # Перелік віджетів
    print("\n" + "=" * 60)
    print("Дочірні віджети:")
    print("=" * 60)
    widgets = tester.list_child_widgets()

    # Зберігаємо результати у файл
    results_dir = Path("tests/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_file = results_dir / "widgets_debug.txt"

    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(f"Вікно: {rect}\n\n")
        for w in widgets:
            f.write(f"hwnd={w['hwnd']} cls={w['class']} text='{w['text']}' rect={w['rect']}\n")

    print(f"\n[OK] Результати збережено: {debug_file}")

    # Скріншот вікна з координатами
    screenshot_path = results_dir / "debug_window.png"
    tester.screenshot_window(str(screenshot_path))

    # Закриття
    tester.close_app()

    print("\n" + "=" * 60)
    print("ДЕБАГ ЗАВЕРШЕНО")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = debug_widgets()
    sys.exit(0 if success else 1)
