"""
Пакетна обробка зображень.
Не залежить від GUI модулів.
GUI викликає методи класу BatchProcessor через сигнали/колбеки.
"""

import os
from typing import Callable
import numpy as np

from core import loader, saver, printer as printer_module
from processing import pipeline
from utils import file_utils


class BatchProcessor:
    """
    Обробляє список файлів у двох режимах:
    - auto:   Auto Fix → друк без втручання
    - manual: повертає зображення одне за одним для перегляду у GUI
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self._files: list[str] = []
        self._index: int = 0

    # ------------------------------------------------------------------
    # Черга файлів
    # ------------------------------------------------------------------

    def set_files(self, paths: list[str]) -> None:
        """Встановлює список файлів для обробки."""
        self._files = file_utils.filter_supported(paths)
        self._index = 0

    def add_files(self, paths: list[str]) -> None:
        """Додає файли до існуючої черги."""
        new = file_utils.filter_supported(paths)
        self._files.extend(new)

    def add_folder(self, folder: str) -> None:
        """Додає всі підтримувані зображення з папки."""
        images = file_utils.collect_images_from_folder(folder)
        self._files.extend(images)

    def clear(self) -> None:
        self._files = []
        self._index = 0

    @property
    def files(self) -> list[str]:
        return list(self._files)

    @property
    def total(self) -> int:
        return len(self._files)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self._index)

    # ------------------------------------------------------------------
    # Авто-режим
    # ------------------------------------------------------------------

    def run_auto(
        self,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
    ) -> int:
        """
        Обробляє всі файли автоматично: Auto Fix → друк.
        on_progress(current, total, filename) — колбек прогресу.
        on_error(filename, message) — колбек помилки.
        Повертає кількість успішно надрукованих файлів.
        """
        s = self.settings
        printed = 0

        for i, path in enumerate(self._files):
            filename = os.path.basename(path)
            if on_progress:
                on_progress(i + 1, self.total, filename)

            try:
                image = loader.load(path)
                processed = pipeline.run_autofix(
                    image,
                    sharpen_strength=s.get("sharpen_strength", 0.4),
                    hdr_strength=s.get("hdr_strength", 0.5),
                    use_hdr=s.get("hdr_in_autofix", True),
                    use_perspective=s.get("auto_perspective", True),
                )
                self._maybe_save(processed, path)
                printer_module.print_image(
                    processed,
                    printer_name=s.get("printer_name", ""),
                    jpg_quality=s.get("jpg_quality", 95),
                )
                printed += 1

            except Exception as exc:
                if on_error:
                    on_error(filename, str(exc))

        self._index = self.total
        return printed

    # ------------------------------------------------------------------
    # Ручний режим — ітератор
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        return self._index < self.total

    def current_file(self) -> str | None:
        if self._index < self.total:
            return self._files[self._index]
        return None

    def load_current(self) -> np.ndarray:
        """Завантажує поточний файл. Кидає RuntimeError якщо черга закінчилась."""
        path = self.current_file()
        if path is None:
            raise RuntimeError("Черга порожня")
        return loader.load(path)

    def print_current(self, processed_image: np.ndarray) -> None:
        """Друкує передане зображення і переходить до наступного."""
        s = self.settings
        path = self.current_file()
        if path:
            self._maybe_save(processed_image, path)
        printer_module.print_image(
            processed_image,
            printer_name=s.get("printer_name", ""),
            jpg_quality=s.get("jpg_quality", 95),
        )
        self._index += 1

    def skip_current(self) -> None:
        """Пропускає поточний файл без друку."""
        self._index += 1

    # ------------------------------------------------------------------
    # Внутрішнє
    # ------------------------------------------------------------------

    def _maybe_save(self, image: np.ndarray, source_path: str) -> None:
        """Зберігає JPG якщо save_folder задано у налаштуваннях."""
        folder = self.settings.get("save_folder", "")
        if not folder:
            return
        out_path = file_utils.build_output_path(source_path, folder)
        quality = self.settings.get("jpg_quality", 95)
        saver.save(image, out_path, quality=quality)
