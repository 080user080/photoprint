"""
Тести для QueueView — перевірка збереження та читання статусу (TODO 4.5).

Покриває:
- get_status для нового елемента (pending)
- get_status після mark_done / mark_error / mark_skipped / mark_current
- get_status для невалідного індексу
- незалежність статусу від тексту елемента
- сценарій _on_auto_done: файли з різними статусами не перезаписуються
"""

import os
import pytest

# Встановлюємо offscreen-платформу ДО імпорту PyQt6, щоб тести працювали без дисплея
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from gui.queue_view import QueueView


# ------------------------------------------------------------------
# Фікстури
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Створює єдиний QApplication для всіх тестів у модулі."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def queue(qapp):
    """Створює QueueView з трьома тестовими файлами."""
    qv = QueueView()
    qv.set_files(["/test/file1.jpg", "/test/file2.jpg", "/test/file3.jpg"])
    yield qv
    qv.deleteLater()


# ------------------------------------------------------------------
# Тести get_status — базові перевірки
# ------------------------------------------------------------------

class TestGetStatus:
    """Тести для методу get_status."""

    def test_new_item_returns_pending(self, queue):
        """Новий елемент має статус 'pending'."""
        assert queue.get_status(0) == "pending"
        assert queue.get_status(1) == "pending"
        assert queue.get_status(2) == "pending"

    def test_after_mark_done(self, queue):
        """Після mark_done статус змінюється на 'done'."""
        queue.mark_done(0)
        assert queue.get_status(0) == "done"

    def test_after_mark_error(self, queue):
        """Після mark_error статус змінюється на 'error'."""
        queue.mark_error(1)
        assert queue.get_status(1) == "error"

    def test_after_mark_skipped(self, queue):
        """Після mark_skipped статус змінюється на 'skipped'."""
        queue.mark_skipped(2)
        assert queue.get_status(2) == "skipped"

    def test_after_mark_current(self, queue):
        """Після mark_current статус змінюється на 'current'."""
        queue.mark_current(0)
        assert queue.get_status(0) == "current"

    def test_invalid_index_returns_pending(self, queue):
        """Невалідний індекс повертає 'pending'."""
        assert queue.get_status(999) == "pending"
        assert queue.get_status(-1) == "pending"

    def test_empty_queue_returns_pending(self, qapp):
        """Порожня черга повертає 'pending' для будь-якого індексу."""
        qv = QueueView()
        assert qv.get_status(0) == "pending"
        qv.deleteLater()


# ------------------------------------------------------------------
# Тести незалежності статусу від тексту
# ------------------------------------------------------------------

class TestStatusIndependentOfText:
    """Тести, що перевіряють незалежність статусу від тексту елемента."""

    def test_done_status_survives_text_change(self, queue):
        """Статус 'done' зберігається після зміни тексту елемента."""
        queue.mark_done(0)
        item = queue.item(0)
        assert item is not None
        item.setText("✓ file1.jpg")
        assert queue.get_status(0) == "done"

    def test_filename_starting_with_checkmark(self, qapp):
        """Файл, чиє ім'я починається з '✓', не повинен хибно визначатись як 'done'."""
        qv = QueueView()
        qv.set_files(["/test/✓ special.jpg"])
        # Статус — pending, хоча текст починається з "✓"
        assert qv.get_status(0) == "pending"
        qv.deleteLater()

    def test_filename_starting_with_x(self, qapp):
        """Файл, чиє ім'я починається з '✗', не повинен хибно визначатись як 'error'."""
        qv = QueueView()
        qv.set_files(["/test/✗ error.jpg"])
        assert qv.get_status(0) == "pending"
        qv.deleteLater()


# ------------------------------------------------------------------
# Тести сценарію _on_auto_done
# ------------------------------------------------------------------

class TestOnAutoDoneScenario:
    """Тести для сценарію _on_auto_done — файли з різними статусами не перезаписуються."""

    def _simulate_on_auto_done(self, queue):
        """Імітація логіки _on_auto_done: позначити всі 'pending' як 'done'."""
        for i in range(queue.count()):
            if queue.get_status(i) not in ("done", "error", "skipped"):
                queue.mark_done(i)

    def test_done_not_overwritten(self, queue):
        """Файл зі статусом 'done' не перезаписується."""
        queue.mark_done(0)
        self._simulate_on_auto_done(queue)
        assert queue.get_status(0) == "done"

    def test_error_not_overwritten(self, queue):
        """Файл зі статусом 'error' не перезаписується."""
        queue.mark_error(1)
        self._simulate_on_auto_done(queue)
        assert queue.get_status(1) == "error"

    def test_skipped_not_overwritten(self, queue):
        """Файл зі статусом 'skipped' не перезаписується."""
        queue.mark_skipped(2)
        self._simulate_on_auto_done(queue)
        assert queue.get_status(2) == "skipped"

    def test_pending_marked_done(self, queue):
        """Файл зі статусом 'pending' позначається як 'done'."""
        queue.mark_done(0)
        queue.mark_error(1)
        self._simulate_on_auto_done(queue)
        # 0 і 1 не змінились
        assert queue.get_status(0) == "done"
        assert queue.get_status(1) == "error"
        # 2 став "done"
        assert queue.get_status(2) == "done"

    def test_all_skipped_not_overwritten(self, queue):
        """Усі файли зі статусом 'skipped' не перезаписуються."""
        queue.mark_skipped(0)
        queue.mark_skipped(1)
        queue.mark_skipped(2)
        self._simulate_on_auto_done(queue)
        assert queue.get_status(0) == "skipped"
        assert queue.get_status(1) == "skipped"
        assert queue.get_status(2) == "skipped"

    def test_mixed_statuses(self, queue):
        """Змішані статуси: done, error, skipped, pending — тільки pending стає done."""
        queue.mark_done(0)
        queue.mark_error(1)
        queue.mark_skipped(2)
        self._simulate_on_auto_done(queue)
        assert queue.get_status(0) == "done"
        assert queue.get_status(1) == "error"
        assert queue.get_status(2) == "skipped"


# ------------------------------------------------------------------
# Тести add_files — статус нових елементів
# ------------------------------------------------------------------

class TestAddFiles:
    """Тести для add_files — перевірка статусу нових елементів."""

    def test_added_file_has_pending_status(self, qapp):
        """Файл, доданий через add_files, має статус 'pending'."""
        qv = QueueView()
        qv.add_files(["/test/new.jpg"])
        assert qv.get_status(0) == "pending"
        qv.deleteLater()

    def test_set_files_all_pending(self, qapp):
        """Усі файли, встановлені через set_files, мають статус 'pending'."""
        qv = QueueView()
        qv.set_files(["/test/a.jpg", "/test/b.jpg", "/test/c.jpg"])
        for i in range(qv.count()):
            assert qv.get_status(i) == "pending"
        qv.deleteLater()