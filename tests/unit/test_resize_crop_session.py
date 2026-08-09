"""Тести переривання hover-crop drag під час resize (Крок 7)."""

import numpy as np
import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPoint, QSize
from PyQt6.QtGui import QResizeEvent

from gui.preview import ImageLabel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_label(qapp):
    label = ImageLabel()
    label.set_image(np.full((80, 100, 3), 128, dtype=np.uint8))
    label.resize(400, 300)
    label.set_crop_rect([
        QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70),
    ])
    return label


def test_resize_aborts_crop_drag_without_commit(qapp):
    label = _make_label(qapp)
    committed = []
    label.crop_session_committed.connect(lambda: committed.append(True))
    label._crop_drag_idx = 0
    label._crop_rect_drag_snapshot = list(label._crop_rect)
    label._crop_session_dirty = True

    label.resizeEvent(QResizeEvent(QSize(420, 320), QSize(400, 300)))

    assert label._crop_drag_idx == -1
    assert label._persp_drag_idx == -1
    assert label._crop_rect == []
    assert label._persp_points == []
    assert label._crop_ready is False
    assert label._crop_session_dirty is False
    assert committed == []

    # Наступні resize-події після першого abort не повинні повторно ламати стан.
    label.resizeEvent(QResizeEvent(QSize(440, 340), QSize(420, 320)))
    assert label._crop_drag_idx == -1
    assert label._persp_drag_idx == -1
    assert committed == []


def test_resize_aborts_perspective_drag_and_clears_session(qapp):
    label = _make_label(qapp)
    label._persp_drag_idx = 1
    label._persp_point_drag_snapshot = QPoint(label._persp_points[1])
    label._persp_detached_drag_snapshot = False
    label._persp_points[1] = QPoint(60, 30)
    label._persp_detached[1] = True

    label._abort_crop_session_due_to_resize()

    assert label._persp_drag_idx == -1
    assert label._persp_point_drag_snapshot is None
    assert label._persp_detached_drag_snapshot is None
    assert label._crop_rect == []
    assert label._persp_points == []
    assert label._persp_detached == [False, False, False, False]


def test_resize_without_drag_keeps_visible_crop_session(qapp):
    label = _make_label(qapp)
    crop_before = list(label._crop_rect)
    label.resizeEvent(QResizeEvent(QSize(420, 320), QSize(400, 300)))

    assert label._crop_drag_idx == -1
    assert label._persp_drag_idx == -1
    assert label._crop_rect == crop_before
    assert label._crop_ready is True
