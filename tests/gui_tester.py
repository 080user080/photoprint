"""Автоматизований тестувальник GUI для PhotoPrint"""
import subprocess
import time
import sys
import threading
from pathlib import Path
from typing import Optional


class GUITester:
    """Базовий клас для автоматизованого тестування GUI через зовнішнє керування"""

    def __init__(self, app_path: str, venv_python: Optional[str] = None):
        self.app_path = app_path  # шлях до скрипта запуску GUI
        self.venv_python = venv_python  # шлях до python у venv (опціонально)
        self.process: Optional[subprocess.Popen] = None  # subprocess process
        self.output_thread: Optional[threading.Thread] = None  # потік для читання виводу
        self.test_images_dir = Path("tests/test_images")
        self.expected_dir = Path("tests/expected")
        self.results_dir = Path("tests/results")
        self.logs_dir = Path("tests/logs")

    def launch_app(self) -> bool:
        """Запускає GUI додаток через subprocess"""
        python = self.venv_python or sys.executable
        try:
            self.process = subprocess.Popen(
                [python, self.app_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(Path(self.app_path).parent)
            )

            # Читання виводу в окремому потоці
            def read_output():
                for line in self.process.stdout:
                    try:
                        if line.strip():
                            print(f"[GUI] {line}", end='')
                    except UnicodeDecodeError:
                        pass  # Пропускаємо помилки кодування

            self.output_thread = threading.Thread(target=read_output, daemon=True)
            self.output_thread.start()
            print(f"[OK] Додаток запущено: {self.app_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Помилка запуску додатка: {e}")
            return False

    def activate_window(self, title: str) -> bool:
        """Активує вікно за назвою (через Windows API)"""
        try:
            import win32gui
            import win32con

            def callback(hwnd, hwnd_list):
                if win32gui.IsWindowVisible(hwnd) and title in win32gui.GetWindowText(hwnd):
                    hwnd_list.append(hwnd)
                return True

            hwnd_list = []
            win32gui.EnumWindows(callback, hwnd_list)

            if hwnd_list:
                hwnd = hwnd_list[0]
                win32gui.SetForegroundWindow(hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                print(f"[OK] Вікно активовано: {title}")
                return True
            else:
                print(f"[ERROR] Вікно не знайдено: {title}")
                return False
        except ImportError:
            print("[ERROR] Встановіть pywin32: pip install pywin32")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка активації вікна: {e}")
            return False

    def type_text(self, text: str) -> bool:
        """Вставляє текст в активне вікно"""
        try:
            import pyautogui
            pyautogui.write(text, interval=0.01)
            print(f"[OK] Текст вставлено: {text}")
            return True
        except ImportError:
            print("[ERROR] Встановіть pyautogui: pip install pyautogui")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка вставки тексту: {e}")
            return False

    def press_key(self, key: str) -> bool:
        """Натискає клавішу"""
        try:
            import pyautogui
            pyautogui.press(key)
            print(f"[OK] Клавіша натиснута: {key}")
            return True
        except ImportError:
            print("[ERROR] Встановіть pyautogui: pip install pyautogui")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка натискання клавіші: {e}")
            return False

    def click_at(self, x: int, y: int) -> bool:
        """Клікає мишкою по координатах"""
        try:
            import pyautogui
            pyautogui.click(x, y)
            print(f"[OK] Клік виконано: ({x}, {y})")
            return True
        except ImportError:
            print("[ERROR] Встановіть pyautogui: pip install pyautogui")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка кліку: {e}")
            return False

    def wait(self, seconds: float) -> None:
        """Чекає заданий час"""
        print(f"[WAIT] Зачекайте {seconds} секунд...")
        time.sleep(seconds)

    def screenshot(self, filename: str) -> bool:
        """Зберігає скріншот вікна"""
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            print(f"[OK] Скріншот збережено: {filename}")
            return True
        except ImportError:
            print("[ERROR] Встановіть pyautogui: pip install pyautogui")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка скріншоту: {e}")
            return False

    def compare_images(self, actual_path: str, expected_path: str, tolerance: int = 5) -> bool:
        """Порівнює два зображення через OpenCV"""
        try:
            import cv2
            import numpy as np

            img1 = cv2.imread(str(actual_path))
            img2 = cv2.imread(str(expected_path))

            if img1 is None or img2 is None:
                print(f"[ERROR] Не вдалося завантажити зображення")
                return False

            # Перевіряємо розміри
            if img1.shape != img2.shape:
                print(f"[ERROR] Розміри зображень не співпадають")
                return False

            # Порівнюємо пікселі з допуском
            diff = np.abs(img1.astype(int) - img2.astype(int))
            max_diff = np.max(diff)

            if max_diff <= tolerance:
                print(f"[OK] Зображення співпадають (max diff: {max_diff})")
                return True
            else:
                print(f"[ERROR] Зображення не співпадають (max diff: {max_diff})")
                return False

        except ImportError:
            print("[ERROR] Встановіть opencv-python: pip install opencv-python")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка порівняння: {e}")
            return False

    def close_app(self) -> None:
        """Закриває GUI додаток"""
        if self.process:
            print("[CLOSE] Закриття програми...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def setup_directories(self) -> None:
        """Створює необхідні директорії"""
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        self.expected_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        print("[OK] Директорії створено")

    def clear_logs(self) -> None:
        """Очищає логи перед тестом"""
        if self.logs_dir.exists():
            for log_file in self.logs_dir.iterdir():
                try:
                    log_file.unlink()
                    print(f"[OK] Видалено лог: {log_file.name}")
                except Exception as e:
                    print(f"[ERROR] Помилка видалення {log_file.name}: {e}")

    def read_logs(self) -> None:
        """Читає логи після тесту"""
        print("\n" + "=" * 60)
        print("ЛОГИ:")
        print("=" * 60)

        if self.logs_dir.exists():
            for log_file in self.logs_dir.iterdir():
                print(f"\n[FILE] {log_file.name}:")
                print("-" * 40)
                try:
                    content = log_file.read_text(encoding='utf-8')
                    if content:
                        print(content)
                    else:
                        print("(пустий)")
                except Exception as e:
                    print(f"[ERROR] Помилка читання {log_file.name}: {e}")
        else:
            print("[ERROR] Папка логів не існує")
