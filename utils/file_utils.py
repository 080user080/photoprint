"""
Утиліти для роботи з файлами.
Не залежить від жодного іншого модуля проєкту.
"""

import os

# Основні формати (завжди доступні через OpenCV/PIL)
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".heic", ".heif",
}

# RAW-формати (потребують rawpy, опціонально)
# Додаємо до SUPPORTED_EXTENSIONS тільки якщо rawpy доступний,
# щоб filter_supported не показував файли, які loader.load не зможе прочитати
_RAW_EXTENSIONS = {
    ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".srw",
    ".raf", ".pef", ".x3f", ".3fr", ".raw",
}
try:
    import rawpy  # noqa: F401
    SUPPORTED_EXTENSIONS.update(_RAW_EXTENSIONS)
except ImportError:
    pass


def is_supported_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def collect_images_from_folder(folder: str) -> list[str]:
    """Повертає список підтримуваних зображень з папки (не рекурсивно)."""
    result = []
    try:
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and is_supported_image(full):
                result.append(full)
    except OSError:
        pass
    return result


def filter_supported(paths: list[str]) -> list[str]:
    """Фільтрує список — залишає тільки підтримувані зображення."""
    return [p for p in paths if is_supported_image(p)]


def build_output_path(source_path: str, save_folder: str, suffix: str = "") -> str:
    """
    Формує шлях для збереження JPG.
    Якщо save_folder порожній — зберігає поруч з оригіналом.
    """
    basename = os.path.splitext(os.path.basename(source_path))[0]
    filename = f"{basename}{suffix}.jpg"
    folder = save_folder if save_folder else os.path.dirname(source_path)
    return os.path.join(folder, filename)
