"""
Скрипт профілювання пам'яті для PhotoPrint.

Проганяє BatchProcessor.run_auto на N тестових зображеннях (N=10, 50, 100)
з використанням tracemalloc, виводить пік пам'яті.

Використання:
  python tools/profile_memory.py                     # N=10 (default)
  python tools/profile_memory.py --count 50
  python tools/profile_memory.py --count 100 --verbose

Залежності (опціонально):
  pip install memory_profiler psutil   # для додаткових метрик
"""

import argparse
import gc
import os
import sys
import tracemalloc
from typing import Optional

# Додаємо корінь проєкту в sys.path, щоб імпорти працювали
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import cv2

from batch.batch_processor import BatchProcessor
from core import loader
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

# Розмір тестового зображення (якщо реальних немає — генеруємо)
TEST_IMAGE_W = 1920
TEST_IMAGE_H = 1080
TEST_IMAGE_CH = 3

# Де шукати реальні тестові зображення
TEST_IMAGES_DIR = os.path.join(_PROJECT_ROOT, "tests", "test_images")

# Налаштування для BatchProcessor (мінімальні, без друку)
BATCH_SETTINGS: dict = {
    "autofix_enabled": True,
    "sharpen_strength": 0.4,
    "hdr_strength": 0.5,
    "hdr_in_autofix": True,
    "auto_perspective": True,
    "bw_binary": False,
    "classify_bw_std_thresh": 20.0,
    "classify_edge_ratio_min": 0.03,
    "classify_line_count_min": 3,
    "shadow_highlight_strength": 0.0,
    "save_folder": "",  # не зберігаємо файли
    "printer_name": "",  # не друкуємо
    "jpg_quality": 95,
}

# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------


def _generate_test_images(count: int, target_dir: str) -> list[str]:
    """
    Генерує `count` синтетичних тестових зображень у `target_dir`.
    Повертає список шляхів до згенерованих файлів.
    Якщо файли вже існують — не перегенеровує.
    """
    os.makedirs(target_dir, exist_ok=True)
    paths: list[str] = []
    for i in range(count):
        path = os.path.join(target_dir, f"profile_test_{i:04d}.jpg")
        if not os.path.isfile(path):
            # Випадкове зображення, схоже на документ
            img = np.random.randint(200, 255, (TEST_IMAGE_H, TEST_IMAGE_W, TEST_IMAGE_CH), dtype=np.uint8)
            # Додаємо "текст" — темні прямокутники
            for _ in range(np.random.randint(5, 15)):
                x1, y1 = np.random.randint(0, TEST_IMAGE_W - 200), np.random.randint(0, TEST_IMAGE_H - 50)
                x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(10, 50)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(path)
    return paths


def _collect_real_test_images() -> list[str]:
    """Збирає реальні тестові зображення з директорії tests/test_images."""
    if not os.path.isdir(TEST_IMAGES_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    images = []
    for fname in sorted(os.listdir(TEST_IMAGES_DIR)):
        _, ext = os.path.splitext(fname)
        if ext.lower() in exts:
            images.append(os.path.join(TEST_IMAGES_DIR, fname))
    return images


def _format_mib(bytes_val: int) -> str:
    """Форматує байти в МіБ з двома знаками після коми."""
    return f"{bytes_val / 1024 / 1024:.2f} MiB"


# ---------------------------------------------------------------------------
# Профілювання
# ---------------------------------------------------------------------------


def profile_batch(
    count: int,
    use_real_images: bool = False,
    verbose: bool = False,
    profile_dir: Optional[str] = None,
) -> dict:
    """
    Проганяє BatchProcessor.run_auto на `count` зображеннях і повертає
    статистику пам'яті.

    Параметри
    ---------
    count : int
        Кількість зображень для обробки.
    use_real_images : bool
        Якщо True — використовує реальні зображення з tests/test_images
        (якщо їх менше ніж count, доповнює синтетичними).
    verbose : bool
        Детальний вивід після кожного файлу.
    profile_dir : str або None
        Якщо вказано — зберігає туди згенеровані тестові зображення
        (за замовчуванням — тимчасова директорія).

    Повертає
    --------
    dict із ключами:
        peak_mb : float
        peak_vm_mb : float (якщо є psutil)
        total_images : int
        duration_sec : float
        avg_per_image_mb : float
        num_gc_collections : int
    """
    import time

    # Підготовка зображень
    if profile_dir is None:
        profile_dir = os.path.join(_PROJECT_ROOT, "tools", "_profile_images")
    os.makedirs(profile_dir, exist_ok=True)

    if use_real_images:
        real = _collect_real_test_images()
        if len(real) >= count:
            images = real[:count]
        else:
            images = real + _generate_test_images(count - len(real), profile_dir)
    else:
        images = _generate_test_images(count, profile_dir)

    if verbose:
        print(f"[profile] Використовується {len(images)} зображень")
        print(f"[profile] Режим: {'реальні + синтетичні' if use_real_images else 'синтетичні'}")

    # Налаштовуємо BatchProcessor — підміняємо files напряму
    bp = BatchProcessor(BATCH_SETTINGS)
    bp._files = images

    # Збираємо сміття перед заміром
    gc.collect()
    gc.collect()
    before_snapshot = tracemalloc.take_snapshot()

    start_time = time.perf_counter()
    gc_count_before = gc.get_count()

    # Скидаємо лічильники tracemalloc
    tracemalloc.reset_peak()

    # Запускаємо
    printed = bp.run_auto(
        on_progress=_make_progress_cb(verbose),
        on_error=_make_error_cb(verbose),
    )

    elapsed = time.perf_counter() - start_time
    gc_count_after = gc.get_count()

    # Після завершення — збираємо сміття ще раз і беремо snapshot
    gc.collect()
    gc.collect()
    after_snapshot = tracemalloc.take_snapshot()

    # Рахуємо різницю
    stats_diff = after_snapshot.compare_to(before_snapshot, "lineno")
    peak_traced = tracemalloc.get_traced_memory()[1]

    # Якщо є psutil — додаємо RSS/VMS
    vm_peak_mb = 0.0
    try:
        import psutil
        proc = psutil.Process()
        vm_peak_mb = proc.memory_info().vms / 1024 / 1024
    except ImportError:
        pass

    result = {
        "peak_traced_mb": peak_traced / 1024 / 1024,
        "peak_vm_mb": vm_peak_mb,
        "total_images": count,
        "duration_sec": round(elapsed, 2),
        "printed": printed,
        "avg_per_image_mb": (peak_traced / count) / 1024 / 1024 if count > 0 else 0.0,
        "gc_collections_diff": tuple(
            a - b for a, b in zip(gc_count_after, gc_count_before)
        ),
    }

    if verbose:
        print(f"\n[profile] Друковано: {printed} / {count}")
        print(f"[profile] Час: {elapsed:.2f} с")
        print(f"[profile] Пік traced memory: {_format_mib(peak_traced)}")
        if vm_peak_mb:
            print(f"[profile] VMS (psutil): {vm_peak_mb:.2f} MiB")
        print(f"[profile] GC збірки (запущено/молоді/середні/старі): {gc_count_after}")
        print(f"\n[profile] Топ-10 за зростанням трасування пам'яті:")
        for stat in stats_diff[:10]:
            print(f"  {stat}")

    return result


def _make_progress_cb(verbose: bool):
    """Повертає callback для on_progress."""
    def _cb(current: int, total: int, filename: str):
        if verbose:
            print(f"  [{current}/{total}] {filename}")
        # Невелика пауза, щоб tracemalloc встиг зафіксувати
        if current % 20 == 0:
            gc.collect()
    return _cb


def _make_error_cb(verbose: bool):
    """Повертає callback для on_error."""
    def _cb(idx: int, filename: str, msg: str):
        if verbose:
            print(f"  [ПОМИЛКА] {filename}: {msg}")
    return _cb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Профілювання пам'яті PhotoPrint")
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Кількість зображень для обробки (10, 50, 100)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Детальний вивід (прогрес, топ-10 трасувань)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Використовувати реальні тестові зображення з tests/test_images",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Прогнати всі N=[10, 50, 100] послідовно",
    )
    args = parser.parse_args()

    # Вмикаємо tracemalloc
    tracemalloc.start(25)

    if args.all:
        print("=" * 60)
        print("ПРОФІЛЮВАННЯ ПАМ'ЯТІ — ВСІ РАУНДИ")
        print("=" * 60)
        results = []
        for n in (10, 50, 100):
            print(f"\n--- N={n} ---")
            r = profile_batch(n, use_real_images=args.real, verbose=args.verbose)
            results.append(r)

        print("\n" + "=" * 60)
        print("ЗВЕДЕННЯ:")
        print("-" * 60)
        print(f"{'N':>6} | {'Peak traced':>14} | {'Peak VMS':>14} | {'Час, с':>8} | {'На 1 зобр.':>12}")
        print("-" * 60)
        for r in results:
            print(
                f"{r['total_images']:>6} | "
                f"{_format_mib(int(r['peak_traced_mb'] * 1024 * 1024)):>14} | "
                f"{r['peak_vm_mb']:>12.2f} MiB | "
                f"{r['duration_sec']:>8.2f} | "
                f"{_format_mib(int(r['avg_per_image_mb'] * 1024 * 1024)):>12}"
            )
        print("=" * 60)
    else:
        print(f"Профілювання пам'яті: N={args.count}")
        print("-" * 40)
        r = profile_batch(args.count, use_real_images=args.real, verbose=args.verbose)
        print("\nРезультат:")
        for k, v in r.items():
            print(f"  {k}: {v}")

    tracemalloc.stop()


if __name__ == "__main__":
    main()