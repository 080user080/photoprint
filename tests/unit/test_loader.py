"""
Юніт-тести для core/loader.py (включно з RAW-форматами).
"""

import numpy as np
import pytest
import os
import tempfile

from core.loader import load, _load_raw, _load_heic, RAW_EXTENSIONS


class TestRAWExtensions:
    """Перевірка, що RAW-розширення визначені."""

    def test_raw_extensions_non_empty(self):
        assert len(RAW_EXTENSIONS) > 0

    def test_raw_extensions_contains_cr2(self):
        assert ".cr2" in RAW_EXTENSIONS

    def test_raw_extensions_contains_nef(self):
        assert ".nef" in RAW_EXTENSIONS

    def test_raw_extensions_contains_arw(self):
        assert ".arw" in RAW_EXTENSIONS

    def test_raw_extensions_contains_dng(self):
        assert ".dng" in RAW_EXTENSIONS


class TestLoadRAW:
    """Тести для _load_raw — без реального RAW-файлу (skipif)."""

    def test_load_raw_nonexistent_file(self):
        """Спроба завантажити неіснуючий RAW-файл → RuntimeError."""
        with pytest.raises(RuntimeError, match="Не вдалося прочитати RAW-файл"):
            _load_raw(r"C:\nonexistent_raw_file.cr2")


class TestLoadGeneral:
    """Загальні тести для loader.load()."""

    def test_load_nonexistent_file(self):
        """Неіснуючий файл → RuntimeError."""
        with pytest.raises(RuntimeError, match="Файл не знайдено"):
            load(r"C:\nonexistent_file_12345.jpg")

    def test_load_invalid_file(self):
        """Файл, що не є зображенням → RuntimeError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not an image data")
            tmp_path = f.name
        try:
            with pytest.raises(RuntimeError, match="Не вдалося декодувати"):
                load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_jpg_success(self, test_image_path):
        """Завантаження реального JPG → BGR uint8."""
        img = load(test_image_path)
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR

    def test_load_jpg_immutability(self, test_image_path):
        """load() не змінює файл на диску (перевірка, що ми не пишемо в оригінал)."""
        before = os.path.getsize(test_image_path)
        load(test_image_path)
        after = os.path.getsize(test_image_path)
        assert before == after


class TestLoadHEIC:
    """Тести для _load_heic — перевірка імпорту та помилок."""

    def test_heic_nonexistent_file(self):
        """Спроба завантажити неіснуючий HEIC → помилка (FileNotFoundError або RuntimeError)."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            _load_heic(r"C:\nonexistent_file.heic")
