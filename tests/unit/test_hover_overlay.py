"""Тести crop-hover без текстового overlay."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QEvent, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

from gui.preview import ImageLabel, PreviewPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def label(qapp):
    lbl = ImageLabel()
    lbl.set_image(np.full((100, 80, 3), 128, dtype=np.uint8))
    lbl.resize(400, 300)
    return lbl


def _inside(label):
    rect = label._img_rect()
    assert rect is not None
    QCursor.setPos(label.mapToGlobal(rect.center()))


def test_hover_enabled_by_default(label):
    assert label._hover_enabled is True


def test_hover_requests_crop_immediately_and_has_no_overlay_timer(label):
    emitted = []
    label.crop_session_requested.connect(lambda: emitted.append(True))
    _inside(label)
    label._maybe_schedule_hover()
    assert emitted == [True]
    assert label._crop_rect_requested_for_current_image is True
    assert label._hover_hide_timer.isActive() is False
    # Зміна курсора навмисно вимкнена за замовчуванням (TODO2: щоб кастомний
    # курсор не заважав перетягуванню хендлів рамки) — лишається стрілка.
    assert label._crop_cursor_enabled is False
    assert label.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_hover_request_is_only_emitted_once_until_reset(label):
    emitted = []
    label.crop_session_requested.connect(lambda: emitted.append(True))
    _inside(label)
    label._maybe_schedule_hover()
    label._maybe_schedule_hover()
    assert len(emitted) == 1
    label._disable_hover()
    label._maybe_schedule_hover()
    assert len(emitted) == 2


def test_leave_schedules_commit_only_for_dirty_session(label):
    label.set_crop_rect([QPoint(0, 0), QPoint(79, 0), QPoint(79, 99), QPoint(0, 99)])
    label._crop_session_dirty = True
    label.leaveEvent(QEvent(QEvent.Type.Leave))
    assert label._hover_hide_timer.isActive() is True


def test_disable_hover_stops_session_timer_and_cursor(label):
    label._hover_hide_timer.start()
    label.set_hover_enabled(False)
    assert label._hover_hide_timer.isActive() is False
    assert label.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_edit_mode_disables_hover(label):
    corners = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
    label.set_edit_mode(True, corners)
    assert label._hover_enabled is False


def test_set_image_and_placeholder_reset_hover(label):
    label._crop_rect_requested_for_current_image = True
    label.set_image(np.full((50, 60, 3), 255, dtype=np.uint8))
    assert label._crop_rect_requested_for_current_image is False
    label._crop_rect_requested_for_current_image = True
    label.set_placeholder()
    assert label._crop_rect_requested_for_current_image is False


def test_cursor_resets_outside_image(label):
    rect = label._img_rect()
    assert rect is not None
    QCursor.setPos(label.mapToGlobal(QPoint(2, 2)))
    label._maybe_schedule_hover()
    assert label.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_crop_cursor_applies_when_enabled(label):
    """Кастомний курсор-кадрування застосовується лише після явного ввімкнення."""
    label.set_crop_cursor_enabled(True)
    _inside(label)
    label._maybe_schedule_hover()
    # Кастомний курсор — QCursor(QPixmap, ...): його pixmap непорожній.
    # У системної стрілки (ArrowCursor) pixmap порожній.
    assert label.cursor().pixmap().isNull() is False


def test_crop_cursor_restores_arrow_when_disabled(label):
    """Вимкнення зміни курсора негайно повертає системну стрілку."""
    label.set_crop_cursor_enabled(True)
    _inside(label)
    label._maybe_schedule_hover()
    assert label.cursor().pixmap().isNull() is False
    label.set_crop_cursor_enabled(False)
    assert label.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_preview_panel_proxy(qapp):
    panel = PreviewPanel()
    panel.set_hover_enabled(False)
    assert panel._before._hover_enabled is False
    panel.set_hover_enabled(True)
    assert panel._before._hover_enabled is True
