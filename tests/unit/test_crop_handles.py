"""
Тести для хендлів прямокутного кадрування (TODO2 крок 4).

Фокус:
1. set_crop_ready / set_crop_rect — керування прапорцем готовності.
2. _disable_hover — скидає _crop_ready і _crop_drag_idx.
3. mousePressEvent — hit-test хендлів кадрування (і стара логіка _edit_mode не ламається).
4. _apply_crop_drag — утримання прямокутної форми + мінімальний розмір.
5. mouseReleaseEvent — скидає _crop_drag_idx і емітить crop_rect_released.
6. leaveEvent — під час drag не скидає стан сесії.
"""

import os
import numpy as np
import pytest

# Встановлюємо offscreen-платформу ДО імпорту PyQt6
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPoint, QPointF, QEvent, Qt
from PyQt6.QtGui import QMouseEvent

from gui.preview import ImageLabel, CROP_MIN_SIZE_PX, CROP_HANDLE_HIT_TOLERANCE


@pytest.fixture(scope="session")
def qapp():
    """Створює єдиний QApplication для всіх тестів."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def label(qapp):
    """ImageLabel з завантаженим зображенням 100x80."""
    img = np.full((80, 100, 3), 128, dtype=np.uint8)
    lbl = ImageLabel()
    lbl.set_image(img)
    lbl.resize(400, 300)
    return lbl


def _make_mouse_event(etype, pos: QPoint) -> QMouseEvent:
    """Створює QMouseEvent для тестів."""
    return QMouseEvent(
        etype,
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )


# ---------------------------------------------------------------------------
# set_crop_ready / set_crop_rect
# ---------------------------------------------------------------------------

class TestCropReady:
    def test_default_false(self, label):
        assert label._crop_ready is False

    def test_set_crop_ready_false(self, label):
        label._crop_ready = True
        label._crop_drag_idx = 2
        label.set_crop_ready(False)
        assert label._crop_ready is False
        assert label._crop_drag_idx == -1

    def test_set_crop_rect_sets_ready(self, label):
        pts = [QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)]
        label.set_crop_rect(pts)
        assert label._crop_ready is True
        assert len(label._crop_rect) == 4

    def test_crop_cursor_is_disabled_by_default(self, label):
        assert label._crop_cursor_enabled is False

    def test_crop_cursor_can_be_toggled_independently(self, label):
        label.set_crop_cursor_enabled(True)
        assert label._crop_cursor_enabled is True
        label.set_crop_cursor_enabled(False)
        assert label._crop_cursor_enabled is False

    def test_disable_hover_resets_crop_state(self, label):
        label.set_crop_rect([QPoint(0, 0), QPoint(10, 0), QPoint(10, 10), QPoint(0, 10)])
        label._crop_drag_idx = 1
        label._disable_hover()
        assert label._crop_ready is False
        assert label._crop_drag_idx == -1
        assert label._crop_rect == []


# ---------------------------------------------------------------------------
# mousePressEvent — hit-test хендлів кадрування
# ---------------------------------------------------------------------------

class TestMousePressCrop:
    def test_no_drag_when_not_ready(self, label):
        # Рамка не готова — клік не повинен почати drag
        label._crop_rect = [QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)]
        label._crop_ready = False
        # Позиція в зоні TL-хендла (widget-координати кута)
        tl_widget = label._img_to_widget(QPoint(10, 10))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_widget)
        label.mousePressEvent(ev)
        assert label._crop_drag_idx == -1

    def test_quick_press_activates_crop_before_hit_testing(self, label):
        label._crop_ready = False
        label.crop_session_requested.connect(lambda: label.set_crop_rect([
            QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70),
        ]))
        tl_widget = label._img_to_widget(QPoint(10, 10))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_widget)

        label.mousePressEvent(ev)

        assert label._crop_ready is True
        assert label._crop_drag_idx == 0

    def test_starts_drag_on_corner_hit(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        tl_widget = label._img_to_widget(QPoint(10, 10))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_widget)
        label.mousePressEvent(ev)
        assert label._crop_drag_idx == 0
        assert label._crop_rect_drag_snapshot == label._crop_rect

    def test_no_drag_outside_handles(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        # Позиція в центрі прямокутника — поза зоною хендлів
        center_widget = label._img_to_widget(QPoint(50, 40))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, center_widget)
        label.mousePressEvent(ev)
        assert label._crop_drag_idx == -1

    def test_edit_mode_still_works(self, label):
        # Стара логіка _edit_mode не повинна ламатись
        label.set_edit_mode(True, [QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        tl_widget = label._img_to_widget(QPoint(10, 10))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_widget)
        label.mousePressEvent(ev)
        # У _edit_mode хіт-тест іде по _points, не по _crop_rect
        assert label._drag_idx == 0
        assert label._crop_drag_idx == -1


# ---------------------------------------------------------------------------
# _apply_crop_drag — утримання прямокутної форми
# ---------------------------------------------------------------------------

class TestApplyCropDrag:
    def _setup(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        return list(label._crop_rect)

    def test_drag_tl_updates_adjacent(self, label):
        self._setup(label)
        label._crop_drag_idx = 0
        label._apply_crop_drag(QPoint(20, 20))
        rect = label._crop_rect
        # TL став (20,20)
        assert rect[0] == QPoint(20, 20)
        # TR ділить Y
        assert rect[1].y() == 20
        # BL ділить X
        assert rect[3].x() == 20
        # BR — якір, не змінився
        assert rect[2] == QPoint(90, 70)
        # Прямокутник лишається осьовирівняним
        assert rect[0].x() == rect[3].x()
        assert rect[0].y() == rect[1].y()
        assert rect[1].x() == rect[2].x()
        assert rect[2].y() == rect[3].y()

    def test_drag_br_updates_adjacent(self, label):
        self._setup(label)
        label._crop_drag_idx = 2
        label._apply_crop_drag(QPoint(80, 60))
        rect = label._crop_rect
        assert rect[2] == QPoint(80, 60)
        # TR ділить X
        assert rect[1].x() == 80
        # BL ділить Y
        assert rect[3].y() == 60
        # TL — якір
        assert rect[0] == QPoint(10, 10)

    def test_min_size_clamped(self, label):
        self._setup(label)
        label._crop_drag_idx = 0
        # Спроба стягнути TL вправо-вниз (зменшити прямокутник нижче мінімуму)
        label._apply_crop_drag(QPoint(95, 65))
        rect = label._crop_rect
        # Ширина і висота не менші за CROP_MIN_SIZE_PX
        width = rect[2].x() - rect[0].x()
        height = rect[2].y() - rect[0].y()
        assert width >= CROP_MIN_SIZE_PX
        assert height >= CROP_MIN_SIZE_PX
        # Точні значення: x_max=90, y_max=70 → затиснуто до 50, 30
        assert rect[0] == QPoint(90 - CROP_MIN_SIZE_PX, 70 - CROP_MIN_SIZE_PX)

    def test_min_size_clamped_br(self, label):
        self._setup(label)
        label._crop_drag_idx = 2
        # Спроба стягнути BR вліво-вгору (зменшити нижче мінімуму)
        label._apply_crop_drag(QPoint(5, 5))
        rect = label._crop_rect
        width = rect[2].x() - rect[0].x()
        height = rect[2].y() - rect[0].y()
        assert width >= CROP_MIN_SIZE_PX
        assert height >= CROP_MIN_SIZE_PX
        # x_min=10, y_min=10 → затиснуто до 50, 50
        assert rect[2] == QPoint(10 + CROP_MIN_SIZE_PX, 10 + CROP_MIN_SIZE_PX)

    def test_emits_crop_rect_changed(self, label):
        self._setup(label)
        label._crop_drag_idx = 0
        emitted = []
        label.crop_rect_changed.connect(lambda pts: emitted.append(pts))
        label._apply_crop_drag(QPoint(20, 20))
        assert len(emitted) == 1
        assert len(emitted[0]) == 4


# ---------------------------------------------------------------------------
# mouseMoveEvent — перетягування через подію
# ---------------------------------------------------------------------------

class TestMouseMoveCrop:
    def test_drag_via_event(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        label._crop_drag_idx = 0
        # Widget-позиція, що відповідає img_pt (20, 20)
        target_widget = label._img_to_widget(QPoint(20, 20))
        ev = _make_mouse_event(QEvent.Type.MouseMove, target_widget)
        label.mouseMoveEvent(ev)
        rect = label._crop_rect
        assert rect[0].x() == 20
        assert rect[0].y() == 20
        assert rect[1].y() == 20
        assert rect[3].x() == 20
        assert rect[2] == QPoint(90, 70)


# ---------------------------------------------------------------------------
# mouseReleaseEvent
# ---------------------------------------------------------------------------

class TestMouseReleaseCrop:
    def test_release_emits_and_resets(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        label._crop_drag_idx = 0
        released = []
        label.crop_rect_released.connect(lambda pts: released.append(pts))
        ev = _make_mouse_event(QEvent.Type.MouseButtonRelease, QPoint(0, 0))
        label.mouseReleaseEvent(ev)
        assert label._crop_drag_idx == -1
        assert len(released) == 1
        assert len(released[0]) == 4

    def test_release_without_drag_no_emit(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        released = []
        label.crop_rect_released.connect(lambda pts: released.append(pts))
        ev = _make_mouse_event(QEvent.Type.MouseButtonRelease, QPoint(0, 0))
        label.mouseReleaseEvent(ev)
        assert len(released) == 0

    def test_touching_crop_corner_without_moving_does_not_commit(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        corner_widget = label._img_to_widget(QPoint(10, 10))
        press = _make_mouse_event(QEvent.Type.MouseButtonPress, corner_widget)
        release = _make_mouse_event(QEvent.Type.MouseButtonRelease, corner_widget)
        released = []
        label.crop_rect_released.connect(lambda pts: released.append(pts))

        label.mousePressEvent(press)
        label.mouseReleaseEvent(release)

        assert released == []
        assert label._crop_session_dirty is False

    def test_release_marks_session_dirty(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        label._crop_drag_idx = 0
        ev = _make_mouse_event(QEvent.Type.MouseButtonRelease, QPoint(0, 0))

        label.mouseReleaseEvent(ev)

        assert label._crop_session_dirty is True

    def test_hide_overlay_emits_commit_once_and_clears_dirty(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        crop_before = list(label._crop_rect)
        persp_before = list(label._persp_points)
        label._crop_session_dirty = True
        committed = []
        label.crop_session_committed.connect(lambda: committed.append(True))

        label._hide_hover_overlay()
        label._hide_hover_overlay()

        assert committed == [True]
        assert label._crop_session_dirty is False
        assert label._crop_rect == crop_before
        assert label._persp_points == persp_before

    def test_hide_overlay_without_changes_does_not_commit(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        committed = []
        label.crop_session_committed.connect(lambda: committed.append(True))

        label._hide_hover_overlay()

        assert committed == []


# ---------------------------------------------------------------------------
# leaveEvent — під час drag не скидає курсор/оверлей
# ---------------------------------------------------------------------------

class TestLeaveEventCrop:
    def test_leave_during_drag_keeps_state(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        label._crop_drag_idx = 0
        # Під час drag leaveEvent не повинен завершувати сесію.
        label.leaveEvent(None)
        assert label._crop_drag_idx == 0

    def test_leave_without_drag_hides(self, label):
        label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
        label._crop_drag_idx = -1
        label._crop_session_dirty = True
        label.leaveEvent(None)
        # Commit hover-сесії відкладається коротким таймером.
        assert label._hover_hide_timer.isActive() is True
