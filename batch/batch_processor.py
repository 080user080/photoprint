"""
Пакетна обробка зображень.
Не залежить від GUI модулів.
GUI не звертається до _index напряму — тільки через публічний API.
"""

import os
from typing import Callable
import numpy as np

from core import loader, saver, printer as printer_module
from processing import pipeline
from utils import file_utils
from utils.logger import get_logger

# Константи для налаштувань за замовчуванням
DEFAULT_SHARPEN_STRENGTH = 0.4
DEFAULT_HDR_STRENGTH = 0.5
DEFAULT_JPG_QUALITY = 95
DEFAULT_CLASSIFY_BW_STD_THRESH = 20.0
DEFAULT_CLASSIFY_EDGE_RATIO_MIN = 0.03
DEFAULT_CLASSIFY_LINE_COUNT_MIN = 3
DEFAULT_SHADOW_HIGHLIGHT_STRENGTH = 0.0


class BatchProcessor:

    def __init__(self, settings: dict):
        self._logger = get_logger(__name__)
        self.settings = settings
        self._files: list[str] = []
        self._index: int = 0

    # ------------------------------------------------------------------
    # Черга
    # ------------------------------------------------------------------

    def set_files(self, paths: list[str]) -> None:
        self._files = file_utils.filter_supported(paths)
        self._index = 0

    def add_files(self, paths: list[str]) -> None:
        new = file_utils.filter_supported(paths)
        # Уникаємо дублікатів
        existing = set(self._files)
        for p in new:
            if p not in existing:
                self._files.append(p)
                existing.add(p)

    def add_folder(self, folder: str) -> None:
        self.add_files(file_utils.collect_images_from_folder(folder))

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
    def current_index(self) -> int:
        """Поточна позиція у черзі (публічний, read-only)."""
        return self._index

    # ------------------------------------------------------------------
    # Авто-режим
    # ------------------------------------------------------------------

    def run_auto(
        self,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_error:    Callable[[int, str, str], None] | None = None,
    ) -> int:
        """
        Обробляє всі файли: Auto Fix → друк.
        on_progress(current_1based, total, filename)
        on_error(index, filename, message)
        Повертає кількість успішно надрукованих.
        """
        s = self.settings
        printed = 0

        for i, path in enumerate(self._files):
            filename = os.path.basename(path)
            if on_progress:
                on_progress(i + 1, self.total, filename)
            try:
                image = loader.load(path)
                if s.get("autofix_enabled", True):
                    processed, _ = pipeline.run_autofix(
                        image,
                        sharpen_strength=s.get("sharpen_strength", DEFAULT_SHARPEN_STRENGTH),
                        hdr_strength=s.get("hdr_strength", DEFAULT_HDR_STRENGTH),
                        use_hdr=s.get("hdr_in_autofix", True),
                        use_perspective=s.get("auto_perspective", True),
                        bw_binary=s.get("bw_binary", False),
                        classify_bw_std_thresh=s.get("classify_bw_std_thresh", DEFAULT_CLASSIFY_BW_STD_THRESH),
                        classify_edge_ratio_min=s.get("classify_edge_ratio_min", DEFAULT_CLASSIFY_EDGE_RATIO_MIN),
                        classify_line_count_min=s.get("classify_line_count_min", DEFAULT_CLASSIFY_LINE_COUNT_MIN),
                        shadow_highlight_strength=s.get("shadow_highlight_strength", DEFAULT_SHADOW_HIGHLIGHT_STRENGTH),
                    )
                else:
                    processed = image
                self._maybe_save(processed, path)
                printer_module.print_image(
                    processed,
                    printer_name=s.get("printer_name", ""),
                    jpg_quality=s.get("jpg_quality", DEFAULT_JPG_QUALITY),
                )
                printed += 1
            except Exception as exc:
                self._logger.error(f"Помилка обробки файлу {filename}: {exc}", exc_info=True)
                if on_error:
                    on_error(i, filename, str(exc))

        self._index = self.total
        return printed

    # ------------------------------------------------------------------
    # Ручний режим
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        return self._index < self.total

    def current_file(self) -> str | None:
        if self._index < self.total:
            return self._files[self._index]
        return None

    def load_current(self) -> np.ndarray:
        path = self.current_file()
        if path is None:
            raise RuntimeError("Черга порожня")
        return loader.load(path)

    def print_current(self, processed_image: np.ndarray) -> str:
        """
        Друкує зображення, зберігає якщо потрібно, переходить до наступного.
        Повертає шлях надрукованого файлу.
        """
        path = self.current_file()
        if path is None:
            raise RuntimeError("Немає поточного файлу")
        s = self.settings
        self._maybe_save(processed_image, path)
        printer_module.print_image(
            processed_image,
            printer_name=s.get("printer_name", ""),
            jpg_quality=s.get("jpg_quality", DEFAULT_JPG_QUALITY),
        )
        self._index += 1
        return path

    def skip_current(self) -> str | None:
        """Пропускає поточний файл. Повертає пропущений шлях."""
        path = self.current_file()
        self._index += 1
        return path

    # ------------------------------------------------------------------
    # Внутрішнє
    # ------------------------------------------------------------------

    def _maybe_save(self, image: np.ndarray, source_path: str) -> None:
        folder = self.settings.get("save_folder", "")
        if not folder:
            return
        # Додаємо суфікс _edited щоб не перезаписувати оригінал
        out_path = file_utils.build_output_path(source_path, folder, suffix="_edited")
        saver.save(image, out_path, quality=self.settings.get("jpg_quality", DEFAULT_JPG_QUALITY))
