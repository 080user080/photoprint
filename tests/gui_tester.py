"""Автоматизований тестувальник GUI для PhotoPrint"""
import json
import subprocess
import time
import sys
import os
import ctypes
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class GUITester:
    """Базовий клас для автоматизованого тестування GUI через зовнішнє керування"""

    # Файли для зберігання геометрії віджетів
    WIDGETS_DEBUG_JSON = Path("tests/results/widgets_debug.json")

    def __init__(self, app_path: str, venv_python: Optional[str] = None,
                 debug_mode: bool = False):
        self.app_path = app_path  # шлях до скрипта запуску GUI
        self.venv_python = venv_python  # шлях до python у venv (опціонально)
        self.process: Optional[subprocess.Popen] = None  # subprocess process
        self.output_thread: Optional[threading.Thread] = None  # потік для читання виводу
        self.test_images_dir = Path("tests/test_images")
        self.expected_dir = Path("tests/expected")
        self.results_dir = Path("tests/results")
        self.logs_dir = Path("tests/logs")
        self._hwnd: Optional[int] = None  # дескриптор вікна
        self.debug_mode = debug_mode
        self._widget_cache: Optional[Dict[str, dict]] = None  # кеш геометрії віджетів

    def launch_app(self) -> bool:
        """Запускає GUI додаток через subprocess"""
        python = self.venv_python or sys.executable
        try:
            env = os.environ.copy()
            if self.debug_mode:
                env["PHOTOPRINT_DEBUG_WIDGETS"] = "1"
            self.process = subprocess.Popen(
                [python, self.app_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(Path(self.app_path).parent),
                env=env,
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

    def activate_window(self, title: str, max_retries: int = 5, retry_delay: float = 1.0) -> bool:
        """Активує вікно за назвою (через Windows API) з повторними спробами"""
        try:
            import win32gui
            import win32con

            for attempt in range(max_retries):
                def callback(hwnd, hwnd_list):
                    if win32gui.IsWindowVisible(hwnd) and title in win32gui.GetWindowText(hwnd):
                        hwnd_list.append(hwnd)
                    return True

                hwnd_list = []
                win32gui.EnumWindows(callback, hwnd_list)

                if hwnd_list:
                    self._hwnd = hwnd_list[0]
                    win32gui.SetForegroundWindow(self._hwnd)
                    win32gui.ShowWindow(self._hwnd, win32con.SW_RESTORE)
                    print(f"[OK] Вікно активовано: {title} (hwnd={self._hwnd})")
                    return True
                else:
                    if attempt < max_retries - 1:
                        print(f"[RETRY] Вікно '{title}' не знайдено, спроба {attempt+1}/{max_retries}...")
                        time.sleep(retry_delay)

            print(f"[ERROR] Вікно не знайдено після {max_retries} спроб: {title}")
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
        self._widget_cache = None

    # ------------------------------------------------------------------
    # Методи для роботи з objectName віджетів
    # ------------------------------------------------------------------

    def load_widget_geometries(self) -> Dict[str, dict]:
        """Завантажує геометрію віджетів з JSON-файлу (debug dump)."""
        if self._widget_cache is not None:
            return self._widget_cache
        json_path = self.WIDGETS_DEBUG_JSON
        if not json_path.exists():
            print(f"[WARN] Файл геометрії віджетів не знайдено: {json_path}")
            print("[INFO] Запустіть додаток з PHOTOPRINT_DEBUG_WIDGETS=1")
            self._widget_cache = {}
            return self._widget_cache
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self._widget_cache = json.load(f)
            print(f"[OK] Завантажено геометрію {len(self._widget_cache)} віджетів")
            return self._widget_cache
        except Exception as e:
            print(f"[ERROR] Помилка читання JSON віджетів: {e}")
            self._widget_cache = {}
            return self._widget_cache

    def get_widget_geometry(self, object_name: str) -> Optional[Dict[str, int]]:
        """Повертає глобальну геометрію віджета за objectName."""
        widgets = self.load_widget_geometries()
        if object_name not in widgets:
            print(f"[WARN] Віджет з objectName='{object_name}' не знайдено")
            print(f"[INFO] Доступні: {', '.join(sorted(widgets.keys()))}")
            return None
        return widgets[object_name]

    def click_widget(self, object_name: str) -> bool:
        """Клікає по центру віджета, знайденого за objectName."""
        geom = self.get_widget_geometry(object_name)
        if geom is None:
            return False
        cx = geom["x"] + geom["width"] // 2
        cy = geom["y"] + geom["height"] // 2
        try:
            import pyautogui
            pyautogui.click(cx, cy)
            print(f"[OK] Клік по віджету '{object_name}': ({cx}, {cy})")
            return True
        except ImportError:
            print("[ERROR] Встановіть pyautogui: pip install pyautogui")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка кліку по віджету '{object_name}': {e}")
            return False

    def get_status_text(self) -> Optional[str]:
        """Повертає текст статусного рядка (objectName='status_label')."""
        # Не використовуємо get_widget_geometry бо текст не зберігається в JSON
        # Використовуємо win32gui для пошуку тексту вікна
        return None

    def wait_for_widget(self, object_name: str, timeout: float = 10.0,
                        poll_interval: float = 0.5) -> Optional[Dict[str, int]]:
        """Чекає поки віджет з'явиться у debug dump."""
        start = time.time()
        self._widget_cache = None  # скидаємо кеш щоб перечитати
        while time.time() - start < timeout:
            geom = self.get_widget_geometry(object_name)
            if geom is not None:
                return geom
            time.sleep(poll_interval)
        print(f"[WARN] Віджет '{object_name}' не з'явився за {timeout}с")
        return None

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

    # ------------------------------------------------------------------
    # Розширені методи для взаємодії з GUI
    # ------------------------------------------------------------------

    def find_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Повертає координати вікна (left, top, right, bottom)"""
        try:
            import win32gui
            if self._hwnd:
                rect = win32gui.GetWindowRect(self._hwnd)
                print(f"[OK] Координати вікна: {rect}")
                return rect
            else:
                print("[ERROR] Вікно не активовано (hwnd=None)")
                return None
        except Exception as e:
            print(f"[ERROR] Помилка отримання координат: {e}")
            return None

    def drag_drop_files(self, file_paths: List[str]) -> bool:
        """Імітує Drag & Drop файлів у вікно додатка через WM_DROPFILES"""
        try:
            import win32gui
            import win32con
            import win32api

            if not self._hwnd:
                print("[ERROR] Вікно не активовано. Спочатку викличте activate_window()")
                return False

            # Створюємо DROPFILES структуру
            # Формат: DROPFILES struct + список шляхів (double-null terminated)
            import struct

            # Кодуємо шляхи в wide string з подвійним null-термінатором
            paths_bytes = b""
            for p in file_paths:
                paths_bytes += p.encode("utf-16-le") + b"\x00\x00"
            paths_bytes += b"\x00\x00"  # подвійний null-термінатор

            # DROPFILES структура: 20 байт
            # pt (POINT x,y) = 8 байт, fNC = 4, fWide = 4
            offset_to_paths = 20  # sizeof(DROPFILES)
            dropfiles = struct.pack(
                "IIIii",
                offset_to_paths,  # offset to file list
                0,                 # pt.x (drop point - не важливо)
                0,                 # pt.y
                0,                 # fNC (не клієнтська область)
                1                  # fWide = 1 (Unicode)
            )

            data = dropfiles + paths_bytes

            # Виділяємо глобальну пам'ять
            GMEM_MOVEABLE = 0x0002
            hGlobal = win32api.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not hGlobal:
                print("[ERROR] Не вдалося виділити пам'ять для DROPFILES")
                return False

            pGlobal = win32api.GlobalLock(hGlobal)
            if not pGlobal:
                print("[ERROR] Не вдалося заблокувати пам'ять")
                win32api.GlobalFree(hGlobal)
                return False

            # Копіюємо дані
            import ctypes
            ctypes.memmove(pGlobal, data, len(data))
            win32api.GlobalUnlock(hGlobal)

            # Відправляємо WM_DROPFILES
            WM_DROPFILES = 0x0233
            win32gui.PostMessage(self._hwnd, WM_DROPFILES, hGlobal, 0)

            print(f"[OK] Drag & Drop виконано: {len(file_paths)} файл(ів)")
            return True

        except ImportError:
            print("[ERROR] Встановіть pywin32: pip install pywin32")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка Drag & Drop: {e}")
            return False

    def click_button_by_text(self, text: str) -> bool:
        """Знаходить кнопку за текстом і клікає на неї (через розрахунок координат для PyQt6)"""
        # Маппінг кнопок до їх позицій (відсотки від розміру вікна)
        # Layout: кнопки в ряд на ~70% висоти вікна, починаючи з ~25% ширини
        # Порядок: Auto Fix, Друк, Пропустити, Друкувати все, Зберегти, Налаштування
        button_map = {
            "auto fix":      (0.28, 0.70),
            "друк":          (0.38, 0.70),
            "пропустити":    (0.48, 0.70),
            "друкувати все": (0.58, 0.70),
            "зберегти":      (0.68, 0.70),
            "налаштування":  (0.78, 0.70),
            # Кнопки управління чергою (ліва панель)
            "додати файли":  (0.07, 0.88),
            "додати папку":  (0.07, 0.92),
            "очистити чергу": (0.07, 0.96),
            # Кнопки перспективи та різне (нижній ряд під слайдерами)
            "авто-перспектива":   (0.32, 0.88),
            "ручна перспектива":  (0.46, 0.88),
            "скинути перспективу": (0.60, 0.88),
            "скинути слайдери":   (0.74, 0.88),
        }

        key = text.lower().strip()
        if key not in button_map:
            print(f"[ERROR] Невідома кнопка: {text}")
            print(f"[INFO] Доступні кнопки: {', '.join(button_map.keys())}")
            return False

        rel_x, rel_y = button_map[key]
        return self._click_relative(rel_x, rel_y, label=f"кнопка '{text}'")

    def set_checkbox(self, text: str, checked: bool = True) -> bool:
        """Знаходить чекбокс за текстом і встановлює його стан (через координати для PyQt6)"""
        checkbox_map = {
            "чорно-білий": (0.27, 0.88),
            "ч/б":         (0.27, 0.88),
        }

        key = text.lower().strip()
        if key not in checkbox_map:
            print(f"[ERROR] Невідомий чекбокс: {text}")
            return False

        rel_x, rel_y = checkbox_map[key]
        return self._click_relative(rel_x, rel_y, label=f"чекбокс '{text}'")

    def _click_relative(self, rel_x: float, rel_y: float, label: str = "") -> bool:
        """Клікає по відносних координатах вікна (0.0-1.0)"""
        rect = self.find_window_rect()
        if not rect:
            return False

        left, top, right, bottom = rect
        win_width = right - left
        win_height = bottom - top

        abs_x = int(left + win_width * rel_x)
        abs_y = int(top + win_height * rel_y)

        try:
            import pyautogui
            pyautogui.click(abs_x, abs_y)
            desc = f" ({label})" if label else ""
            print(f"[OK] Клік по відносних координатах{desc}: ({rel_x:.2f}, {rel_y:.2f}) -> ({abs_x}, {abs_y})")
            return True
        except Exception as e:
            print(f"[ERROR] Помилка кліку: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """Натискає комбінацію клавіш (наприклад hotkey('ctrl', 's'))"""
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            print(f"[OK] Комбінація клавіш: {'+'.join(keys)}")
            return True
        except ImportError:
            print("[ERROR] Встановіть pyautogui: pip install pyautogui")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка комбінації клавіш: {e}")
            return False

    def drag_slider(self, slider_label: str, value: float) -> bool:
        """Встановлює значення слайдера за назвою (через відносні координати вікна)"""
        # Маппінг слайдерів до позицій (rel_y, row, start_x_fraction)
        # Ряд 1 (y~0.78): Тіні, Яскравість, Контраст
        # Ряд 2 (y~0.82): Різкість, HDR
        slider_map = {
            "тіні":        {"y": 0.78, "x_start": 0.22, "x_end": 0.36},
            "тени":        {"y": 0.78, "x_start": 0.22, "x_end": 0.36},
            "shadow":      {"y": 0.78, "x_start": 0.22, "x_end": 0.36},
            "яскравість":  {"y": 0.78, "x_start": 0.38, "x_end": 0.52},
            "brightness":  {"y": 0.78, "x_start": 0.38, "x_end": 0.52},
            "контраст":    {"y": 0.78, "x_start": 0.54, "x_end": 0.68},
            "contrast":    {"y": 0.78, "x_start": 0.54, "x_end": 0.68},
            "різкість":    {"y": 0.82, "x_start": 0.22, "x_end": 0.36},
            "різкість":   {"y": 0.82, "x_start": 0.22, "x_end": 0.36},
            "sharpness":   {"y": 0.82, "x_start": 0.22, "x_end": 0.36},
            "sharpen":     {"y": 0.82, "x_start": 0.22, "x_end": 0.36},
            "hdr":         {"y": 0.82, "x_start": 0.38, "x_end": 0.52},
        }

        key = slider_label.lower().strip()
        if key not in slider_map:
            print(f"[ERROR] Невідомий слайдер: {slider_label}")
            print(f"[INFO] Доступні слайдери: {', '.join(set(slider_map.keys()))}")
            return False

        info = slider_map[key]
        # Нормалізуємо value до 0.0-1.0 діапазону слайдера
        # Для слайдерів з від'ємними значеннями (яскравість, контраст) - value 0.5 = нейтральне
        rel_x = info["x_start"] + (info["x_end"] - info["x_start"]) * value
        rel_y = info["y"]

        return self._click_relative(rel_x, rel_y, label=f"слайдер '{slider_label}'={value}")

    def screenshot_window(self, filename: str) -> bool:
        """Зберігає скріншот тільки вікна додатка"""
        try:
            import win32gui
            import pyautogui
            from PIL import Image

            if not self._hwnd:
                print("[ERROR] Вікно не активовано")
                return False

            rect = win32gui.GetWindowRect(self._hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top

            # Робимо повний скріншот і обрізаємо
            screenshot = pyautogui.screenshot()
            cropped = screenshot.crop((left, top, right, bottom))
            cropped.save(filename)
            print(f"[OK] Скріншот вікна збережено: {filename} ({width}x{height})")
            return True

        except ImportError:
            print("[ERROR] Встановіть pyautogui, Pillow, pywin32")
            return False
        except Exception as e:
            print(f"[ERROR] Помилка скріншоту вікна: {e}")
            return False

    def get_window_text(self) -> Optional[str]:
        """Повертає заголовок вікна"""
        try:
            import win32gui
            if self._hwnd:
                return win32gui.GetWindowText(self._hwnd)
            return None
        except Exception:
            return None

    def list_child_widgets(self) -> List[dict]:
        """Перелічує всі дочірні віджети вікна (для дебагу)"""
        try:
            import win32gui
            widgets = []

            def enum_child_callback(hwnd, _):
                cls = win32gui.GetClassName(hwnd)
                text = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                widgets.append({
                    "hwnd": hwnd,
                    "class": cls,
                    "text": text,
                    "rect": rect,
                })
                return True

            if self._hwnd:
                win32gui.EnumChildWindows(self._hwnd, enum_child_callback, None)

            print(f"[OK] Знайдено {len(widgets)} дочірніх віджетів")
            for w in widgets:
                print(f"  hwnd={w['hwnd']} cls={w['class']} text='{w['text']}' rect={w['rect']}")

            return widgets

        except ImportError:
            print("[ERROR] Встановіть pywin32: pip install pywin32")
            return []
        except Exception as e:
            print(f"[ERROR] Помилка переліку віджетів: {e}")
            return []

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
