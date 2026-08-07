"""
Тести для hover-оверлея ImageLabel (TODO2 крок 2).

Фокус:
1. Hover-таймер: 100 мс затримка показу оверлея, 50 мс приховування.
2. set_hover_enabled(False) — негайно ховає оверлей і зупиняє таймери.
3. Активний edit_mode (стара перспектива) — hover не показується.
4. set_image/set_placeholder скидають hover-стан.
5. PreviewPanel.set_hover_enabled проксі до _before.
6. Курсор-кадрування (TODO2 крок 2.3).
7. Таймер не перезапускається, якщо вже активний (TODO2 крок 2.1).
"""

import os
import numpy as np
import pytest

# Встановлюємо offscreen-платформу ДО імпорту PyQt6, щоб тести працювали без дисплея
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QEvent
from PyQt6.QtGui import QCursor

from gui.preview import ImageLabel, PreviewPanel


@pytest.fixture(scope="session")
def qapp():
    """Створює єдиний QApplication для всіх тестів (сумісно з test_queue_view.py)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def label(qapp):
    """ImageLabel з завантаженим зображенням (маленьке, щоб _img_rect() був не None)."""
    img = np.full((100, 80, 3), 128, dtype=np.uint8)
    lbl = ImageLabel()
    lbl.set_image(img)
    lbl.resize(400, 300)
    return lbl


# ---------------------------------------------------------------------------
# Базовий hover-стан
# ---------------------------------------------------------------------------

class TestHoverStateInitial:
    def test_hover_enabled_by_default(self, label):
        assert label._hover_enabled is True

    def test_hover_not_visible_initially(self, label):
        assert label._hover_visible is False

    def test_img_rect_present(self, label):
        rect = label._img_rect()
        assert rect is not None


# ---------------------------------------------------------------------------
# Показ та приховування оверлея через таймери
# ---------------------------------------------------------------------------

class TestHoverShowHideTimers:
    def test_crop_session_is_requested_before_overlay_is_visible(self, label):
        label.crop_session_requested.connect(lambda: label.set_crop_rect([
            QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70),
        ]))
        rect = label._img_rect()
        assert rect is not None
        QCursor.setPos(label.mapToGlobal(rect.center()))

        label._maybe_schedule_hover()

        assert label._crop_ready is True
        assert label._hover_visible is False
        assert label._hover_timer.isActive() is True

    def test_show_timer_starts(self, label):
        # Встановлюємо курсор всередину _img_rect() (глобальна позиція)
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label._hover_timer.isActive()

    def test_timer_not_restarted_when_active(self, label):
        """TODO2 крок 2.1: повторний виклик не перезапускає вже активний таймер."""
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label._hover_timer.isActive()
        # Запам'ятовуємо час, що лишився
        remaining = label._hover_timer.remainingTime()
        # Повторний виклик — таймер не повинен перезапуститись
        label._maybe_schedule_hover()
        assert label._hover_timer.isActive()
        assert label._hover_timer.remainingTime() <= remaining

    def test_show_timer_calls_show(self, label):
        label._show_hover_overlay()
        assert label._hover_visible is True

    def test_hide_timer_calls_hide(self, label):
        label._hover_visible = True
        label._hide_hover_overlay()
        assert label._hover_visible is False

    def test_show_when_edit_mode_does_nothing(self, label):
        corners = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
        label.set_edit_mode(True, corners)
        label._show_hover_overlay()
        assert label._hover_visible is False


# ---------------------------------------------------------------------------
# set_hover_enabled
# ---------------------------------------------------------------------------

class TestHoverEnabled:
    def test_disable_stops_timers_and_hides(self, label):
        label._hover_timer.start()
        label._hover_hide_timer.start()
        label._hover_visible = True
        label.set_hover_enabled(False)
        assert label._hover_enabled is False
        assert label._hover_visible is False
        assert label._hover_timer.isActive() is False
        assert label._hover_hide_timer.isActive() is False

    def test_enable_does_not_hide(self, label):
        label._hover_visible = True
        label.set_hover_enabled(True)
        assert label._hover_visible is True

    def test_disable_prevents_show(self, label):
        label.set_hover_enabled(False)
        label._show_hover_overlay()
        assert label._hover_visible is False


# ---------------------------------------------------------------------------
# Взаємодія з edit_mode та set_image/set_placeholder
# ---------------------------------------------------------------------------

class TestHoverWithModes:
    def test_edit_mode_disables_hover(self, label):
        corners = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
        label.set_edit_mode(True, corners)
        assert label._hover_enabled is False

    def test_set_image_resets_hover(self, label):
        label._hover_visible = True
        label._hover_timer.start()
        img = np.full((50, 60, 3), 255, dtype=np.uint8)
        label.set_image(img)
        assert label._hover_visible is False
        assert label._hover_timer.isActive() is False

    def test_set_placeholder_resets_hover(self, label):
        label._hover_visible = True
        label._hover_timer.start()
        label.set_placeholder()
        assert label._hover_visible is False
        assert label._hover_timer.isActive() is False


# ---------------------------------------------------------------------------
# Курсор-кадрування (TODO2 крок 2.3)
# ---------------------------------------------------------------------------

class TestCropCursor:
    def test_cursor_changes_inside_img_rect(self, label):
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() != Qt.CursorShape.ArrowCursor

    def test_cursor_resets_outside_img_rect(self, label):
        # Курсор поза _img_rect(), але всередині віджета
        rect = label._img_rect()
        assert rect is not None
        # Точка поза зображенням (кут віджета)
        outside_pos = label.mapToGlobal(QPoint(2, 2))
        QCursor.setPos(outside_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor

    def test_cursor_not_changed_when_hover_disabled(self, label):
        label.set_hover_enabled(False)
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor

    def test_cursor_not_changed_in_edit_mode(self, label):
        corners = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
        label.set_edit_mode(True, corners)
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor

    def test_cursor_reset_on_leave(self, label):
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() != Qt.CursorShape.ArrowCursor
        # Симулюємо вихід (leaveEvent приймає QEvent)
        label.leaveEvent(QEvent(QEvent.Type.Leave))
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor

    def test_cursor_reset_on_set_hover_enabled_false(self, label):
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() != Qt.CursorShape.ArrowCursor
        label.set_hover_enabled(False)
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor

    def test_crop_cursor_cached(self, label):
        c1 = label._get_crop_cursor()
        c2 = label._get_crop_cursor()
        assert c1 is c2

    def test_crop_cursor_is_qcursor(self, label):
        c = label._get_crop_cursor()
        assert isinstance(c, QCursor)

    def test_cursor_reset_on_set_image(self, label):
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() != Qt.CursorShape.ArrowCursor
        img = np.full((50, 60, 3), 255, dtype=np.uint8)
        label.set_image(img)
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor

    def test_cursor_reset_on_set_placeholder(self, label):
        rect = label._img_rect()
        assert rect is not None
        global_pos = label.mapToGlobal(rect.center())
        QCursor.setPos(global_pos)
        label._maybe_schedule_hover()
        assert label.cursor().shape() != Qt.CursorShape.ArrowCursor
        label.set_placeholder()
        assert label.cursor().shape() == Qt.CursorShape.ArrowCursor


# ---------------------------------------------------------------------------
# PreviewPanel проксі
# ---------------------------------------------------------------------------

class TestPreviewPanelProxy:
    def test_set_hover_enabled_propagates_to_before(self, qapp):
        panel = PreviewPanel()
        assert panel._before._hover_enabled is True
        panel.set_hover_enabled(False)
        assert panel._before._hover_enabled is False
        panel.set_hover_enabled(True)
        assert panel._before._hover_enabled is True

    def test_disable_perspective_edit_reenables_hover(self, qapp):
        corners = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
        panel = PreviewPanel()
        panel.enable_perspective_edit(corners)
        assert panel._before._hover_enabled is False
        panel.disable_perspective_edit()
        assert panel._before._hover_enabled is True

    def test_clear_disables_hover(self, qapp):
        panel = PreviewPanel()
        panel._before._hover_visible = True
        panel._before._hover_timer.start()
        panel.clear()
        assert panel._before._hover_visible is False
        assert panel._before._hover_timer.isActive() is False

    def test_hover_enabled_on_after_label_not_affected(self, qapp):
        panel = PreviewPanel()
        panel.set_hover_enabled(False)
        assert panel._after._hover_enabled is True
