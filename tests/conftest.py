"""
conftest — додає шлях до кореня проєкту для імпорту processing/* та core/*.
"""
import sys
import os
import pytest

# Додаємо корінь photoprint-main до sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture
def test_image_path() -> str:
    """Повертає шлях до тестового JPG-зображення."""
    return os.path.join(_project_root, "tests", "test_images", "001.jpg")


# Маркер для GUI-тестів, що потребують реальний desktop
def pytest_configure(config):
    config.addinivalue_line("markers", "gui: Тести, що потребують реальний desktop з GUI (Windows)")


@pytest.fixture
def gui_tester():
    """Фікстура для GUITester з desktop-режимом ( реальний Windows desktop)."""
    from tests.gui_tester import GUITester
    tester = GUITester("main.py", debug_mode=False)
    tester.setup_directories()
    yield tester
    tester.close_app()
