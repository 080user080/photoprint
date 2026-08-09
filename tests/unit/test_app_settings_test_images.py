"""
Тести для налаштувань тестових зображень (автозавантаження при старті).
Перевіряємо round-trip app_settings.load()/save() для нових ключів секції [tests].
"""

import os
import tempfile

from config import app_settings


def _make_tmp_ini() -> str:
    """Створює тимчасовий шлях до .ini у системній temp-папці."""
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    return path


def test_defaults_when_no_ini_file():
    """Без файлу — нові ключі мають дефолтні значення."""
    path = os.path.join(tempfile.mkdtemp(), "missing.ini")
    s = app_settings.load(path)
    assert s["test_images_enabled"] is False
    assert s["test_images_folder"] == ""


def test_roundtrip_enabled_true_and_custom_folder():
    """Збереження і повторне читання увімкнених тестових зображень зі своєю папкою."""
    path = _make_tmp_ini()
    try:
        settings = {"test_images_enabled": True, "test_images_folder": r"D:\some\folder"}
        app_settings.save(settings, path)
        loaded = app_settings.load(path)
        assert loaded["test_images_enabled"] is True
        assert loaded["test_images_folder"] == r"D:\some\folder"
    finally:
        os.unlink(path)


def test_roundtrip_disabled_and_empty_folder():
    """Збереження і повторне читання вимкнених зображень з порожньою папкою."""
    path = _make_tmp_ini()
    try:
        settings = {"test_images_enabled": False, "test_images_folder": ""}
        app_settings.save(settings, path)
        loaded = app_settings.load(path)
        assert loaded["test_images_enabled"] is False
        assert loaded["test_images_folder"] == ""
    finally:
        os.unlink(path)


def test_roundtrip_ignores_missing_keys():
    """Якщо ключі відсутні у словнику — save не падає, load повертає дефолти."""
    path = _make_tmp_ini()
    try:
        app_settings.save({}, path)
        loaded = app_settings.load(path)
        assert loaded["test_images_enabled"] is False
        assert loaded["test_images_folder"] == ""
    finally:
        os.unlink(path)