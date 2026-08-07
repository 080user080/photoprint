"""
Тести для дефолтної стартової рамки кадрування (TODO2 крок 3).

Фокус:
1. _full_frame_crop_corners — кути точно по межах зображення.
2. _invalidate_crop_default_cache — скидає кеш.
3. _on_crop_session_requested — з валідним кешем не запускає детекцію повторно.
4. _on_crop_session_requested — з _base is None — no-op.
5. ImageLabel.set_crop_rect — зберігає точки і викликає update.
6. ImageLabel._crop_rect_requested_for_current_image — скидається в _disable_hover.
7. ImageLabel._show_hover_overlay — емітить crop_session_requested один раз.
8. PreviewPanel.crop_session_requested — проксі-сигнал.
"""

import os
import numpy as np
import pytest

# Встановлюємо offscreen-платформу ДО імпорту PyQt6
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QCursor

from gui.preview import ImageLabel, PreviewPanel


@pytest.fixture(scope="session")
def qapp():
    """Створює єдиний QApplication для всіх тестів."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def label(qapp):
    """ImageLabel з завантаженим зображенням."""
    img = np.full((100, 80, 3), 128, dtype=np.uint8)
    lbl = ImageLabel()
    lbl.set_image(img)
    lbl.resize(400, 300)
    return lbl


def _make_window():
    """Створює MainWindow без QApplication (тільки для чистих методів)."""
    from gui.main_window import MainWindow
    return MainWindow.__new__(MainWindow)


# ---------------------------------------------------------------------------
# _full_frame_crop_corners
# ---------------------------------------------------------------------------

class TestFullFrameCropCorners:
    def test_returns_exact_frame_bounds(self):
        w = _make_window()
        img = np.zeros((50, 40, 3), dtype=np.uint8)
        corners = w._full_frame_crop_corners(img)
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32
        expected = np.array([
            [0, 0],
            [39, 0],
            [39, 49],
            [0, 49],
        ], dtype=np.float32)
        np.testing.assert_array_equal(corners, expected)

    def test_returns_exact_frame_bounds_small(self):
        w = _make_window()
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        corners = w._full_frame_crop_corners(img)
        expected = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ], dtype=np.float32)
        np.testing.assert_array_equal(corners, expected)

    def test_returns_exact_frame_bounds_wide(self):
        w = _make_window()
        img = np.zeros((10, 100, 3), dtype=np.uint8)
        corners = w._full_frame_crop_corners(img)
        expected = np.array([
            [0, 0],
            [99, 0],
            [99, 9],
            [0, 9],
        ], dtype=np.float32)
        np.testing.assert_array_equal(corners, expected)


# ---------------------------------------------------------------------------
# _invalidate_crop_default_cache
# ---------------------------------------------------------------------------

class TestInvalidateCropDefaultCache:
    def test_clears_cache(self):
        w = _make_window()
        w._crop_default_corners_full = np.zeros((4, 2), dtype=np.float32)
        w._crop_default_corners_base_id = 123
        w._invalidate_crop_default_cache()
        assert w._crop_default_corners_full is None
        assert w._crop_default_corners_base_id is None

    def test_idempotent(self):
        w = _make_window()
        w._crop_default_corners_full = None
        w._crop_default_corners_base_id = None
        w._invalidate_crop_default_cache()
        assert w._crop_default_corners_full is None
        assert w._crop_default_corners_base_id is None


# ---------------------------------------------------------------------------
# _on_crop_session_requested
# ---------------------------------------------------------------------------

class TestOnCropSessionRequested:
    def test_noop_when_base_none(self):
        w = _make_window()
        w._base = None
        w._crop_detection_in_progress = False
        # Не повинно кидати виняток і не запускати детекцію
        w._on_crop_session_requested()
        assert w._crop_detection_in_progress is False

    def test_uses_cache_without_detection(self):
        w = _make_window()
        base = np.zeros((50, 40, 3), dtype=np.uint8)
        w._base = base
        w._crop_default_corners_full = np.array([
            [0, 0], [39, 0], [39, 49], [0, 49]
        ], dtype=np.float32)
        w._crop_default_corners_base_id = id(base)
        w._crop_detection_in_progress = False
        # Мок preview._before.set_crop_rect
        calls = []
        w._preview = type("P", (), {
            "_before": type("B", (), {
                "set_crop_rect": lambda self, pts: calls.append(pts)
            })()
        })()
        w._corners_to_preview_pts = lambda corners, source: [QPoint(int(c[0]), int(c[1])) for c in corners]

        w._on_crop_session_requested()

        assert len(calls) == 1
        assert len(calls[0]) == 4
        # Детекція не запускалась
        assert w._crop_detection_in_progress is False

    def test_uses_full_frame_immediately_when_cache_invalid(self):
        w = _make_window()
        base = np.zeros((50, 40, 3), dtype=np.uint8)
        w._base = base
        w._crop_default_corners_full = None
        w._crop_default_corners_base_id = None
        w._crop_detection_in_progress = False
        calls = []
        w._preview = type("P", (), {
            "_before": type("B", (), {
                "set_crop_rect": lambda self, pts: calls.append(pts),
            })()
        })()
        w._corners_to_preview_pts = lambda corners, source: [
            QPoint(int(c[0]), int(c[1])) for c in corners
        ]

        w._on_crop_session_requested()

        assert w._crop_detection_in_progress is False
        assert len(calls) == 1
        assert calls[0][0] == QPoint(0, 0)
        assert calls[0][2] == QPoint(39, 49)

# ---------------------------------------------------------------------------
# ImageLabel.set_crop_rect
# ---------------------------------------------------------------------------

class TestSetCropRect:
    def test_stores_points(self, label):
        pts = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
        label.set_crop_rect(pts)
        assert len(label._crop_rect) == 4
        assert label._crop_rect[0] == QPoint(0, 0)
        assert label._crop_rect[3] == QPoint(0, 10)

    def test_clears_previous(self, label):
        label.set_crop_rect([QPoint(0, 0), QPoint(1, 0), QPoint(1, 1), QPoint(0, 1)])
        label.set_crop_rect([QPoint(5, 5), QPoint(6, 5), QPoint(6, 6), QPoint(5, 6)])
        assert len(label._crop_rect) == 4
        assert label._crop_rect[0] == QPoint(5, 5)

    def test_empty_clears(self, label):
        label.set_crop_rect([QPoint(0, 0), QPoint(1, 0), QPoint(1, 1), QPoint(0, 1)])
        label.set_crop_rect([])
        assert label._crop_rect == []


# ---------------------------------------------------------------------------
# _crop_rect_requested_for_current_image
# ---------------------------------------------------------------------------

class TestCropRectRequestedFlag:
    def test_reset_on_disable_hover(self, label):
        label._crop_rect_requested_for_current_image = True
        label._disable_hover()
        assert label._crop_rect_requested_for_current_image is False

    def test_reset_on_set_image(self, label):
        label._crop_rect_requested_for_current_image = True
        img = np.full((50, 60, 3), 255, dtype=np.uint8)
        label.set_image(img)
        assert label._crop_rect_requested_for_current_image is False

    def test_reset_on_set_placeholder(self, label):
        label._crop_rect_requested_for_current_image = True
        label.set_placeholder()
        assert label._crop_rect_requested_for_current_image is False

    def test_show_hover_emits_once(self, label):
        emitted = []
        label.crop_session_requested.connect(lambda: emitted.append(True))
        rect = label._img_rect()
        QCursor.setPos(label.mapToGlobal(rect.center()))
        label._maybe_schedule_hover()
        label._show_hover_overlay()
        label._show_hover_overlay()
        assert len(emitted) == 1
        assert label._crop_rect_requested_for_current_image is True

    def test_show_hover_emits_after_reset(self, label):
        emitted = []
        label.crop_session_requested.connect(lambda: emitted.append(True))
        rect = label._img_rect()
        QCursor.setPos(label.mapToGlobal(rect.center()))
        label._maybe_schedule_hover()
        label._disable_hover()
        label._maybe_schedule_hover()
        assert len(emitted) == 2


# ---------------------------------------------------------------------------
# PreviewPanel проксі-сигнал
# ---------------------------------------------------------------------------

class TestPreviewPanelCropSignal:
    def test_crop_session_requested_propagates(self, qapp):
        panel = PreviewPanel()
        emitted = []
        panel.crop_session_requested.connect(lambda: emitted.append(True))
        panel._before.set_image(np.full((50, 60, 3), 128, dtype=np.uint8))
        rect = panel._before._img_rect()
        QCursor.setPos(panel._before.mapToGlobal(rect.center()))
        panel._before._maybe_schedule_hover()
        assert len(emitted) == 1
