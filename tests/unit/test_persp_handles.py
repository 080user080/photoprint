"""
Тести для хендлів перспективи-кружечків (TODO2 крок 5).

Фокус:
1. set_crop_rect — скидає _persp_detached і ініціалізує _persp_points як похідні.
2. _compute_linked_persp_point / _sync_linked_persp_points — обчислення похідної позиції.
3. _disable_hover — скидає весь стан перспективи.
4. mousePressEvent — hit-test кружечків (пріоритет над хендлами кадрування).
5. mouseMoveEvent — незалежний drag кружечка.
6. mouseReleaseEvent — емітить persp_points_changed_hover.
7. _apply_crop_drag — прив'язані кружечки слідують за рамкою, відв'язані — ні.
8. leaveEvent — під час drag кружечка не скидає стан.
"""

import os
import numpy as np
import pytest

# Встановлюємо offscreen-платформу ДО імпорту PyQt6
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPoint, QPointF, QEvent, Qt
from PyQt6.QtGui import QMouseEvent

from gui.preview import (
    ImageLabel,
    PERSP_HANDLE_INSET_RATIO,
    PERSP_HANDLE_HIT_TOLERANCE,
    CROP_HANDLE_HIT_TOLERANCE,
)


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


def _setup_crop(label):
    """Встановлює прямокутну рамку кадрування і повертає її."""
    label.set_crop_rect([QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)])
    return list(label._crop_rect)


# ---------------------------------------------------------------------------
# set_crop_rect — ініціалізація стану перспективи
# ---------------------------------------------------------------------------

class TestSetCropRectPersp:
    def test_initializes_persp_points(self, label):
        _setup_crop(label)
        assert len(label._persp_points) == 4
        # Кружечки зміщені всередину від кутів
        assert label._persp_points[0].x() > 10
        assert label._persp_points[0].y() > 10
        assert label._persp_points[2].x() < 90
        assert label._persp_points[2].y() < 70

    def test_resets_detached(self, label):
        _setup_crop(label)
        # Штучно відв'язуємо всі
        label._persp_detached = [True, True, True, True]
        # Нова рамка — скидає detached
        label.set_crop_rect([QPoint(0, 0), QPoint(50, 0), QPoint(50, 50), QPoint(0, 50)])
        assert label._persp_detached == [False, False, False, False]

    def test_resets_drag_idx(self, label):
        _setup_crop(label)
        label._persp_drag_idx = 2
        label.set_crop_rect([QPoint(0, 0), QPoint(50, 0), QPoint(50, 50), QPoint(0, 50)])
        assert label._persp_drag_idx == -1


# ---------------------------------------------------------------------------
# _compute_linked_persp_point / _sync_linked_persp_points
# ---------------------------------------------------------------------------

class TestLinkedPerspPoint:
    def test_compute_linked_point(self, label):
        _setup_crop(label)
        # Рамка: TL(10,10), TR(90,10), BR(90,70), BL(10,70)
        # Центр: (50, 40)
        # TL-кружечок: 10 + (50-10)*0.2 = 18, 10 + (40-10)*0.2 = 16
        pt = label._compute_linked_persp_point(0)
        assert pt == QPoint(18, 16)

    def test_sync_does_not_touch_detached(self, label):
        _setup_crop(label)
        # Відв'язуємо TL (індекс 0) і ставимо йому довільну позицію
        label._persp_detached[0] = True
        label._persp_points[0] = QPoint(5, 5)
        # Змінюємо рамку
        label._crop_rect = [QPoint(0, 0), QPoint(50, 0), QPoint(50, 50), QPoint(0, 50)]
        label._sync_linked_persp_points()
        # Відв'язаний кружечок не змінився
        assert label._persp_points[0] == QPoint(5, 5)
        # Решта перерахувались
        assert label._persp_points[1] != QPoint(5, 5)


# ---------------------------------------------------------------------------
# _disable_hover — скидання стану перспективи
# ---------------------------------------------------------------------------

class TestDisableHoverPersp:
    def test_resets_persp_state(self, label):
        _setup_crop(label)
        label._persp_detached = [True, False, True, False]
        label._persp_drag_idx = 1
        label._disable_hover()
        assert label._persp_points == []
        assert label._persp_detached == [False, False, False, False]
        assert label._persp_drag_idx == -1


# ---------------------------------------------------------------------------
# mousePressEvent — hit-test кружечків
# ---------------------------------------------------------------------------

class TestMousePressPersp:
    def test_starts_drag_on_persp_hit(self, label):
        _setup_crop(label)
        # Кружечок TL у widget-координатах
        tl_persp_widget = label._img_to_widget(label._persp_points[0])
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_persp_widget)
        label.mousePressEvent(ev)
        assert label._persp_drag_idx == 0
        # Просте натискання ще не від'єднує кружечок.
        assert label._persp_detached[0] is False
        assert label._persp_point_drag_snapshot is not None
        assert label._persp_detached_drag_snapshot is False

    def test_persp_priority_over_crop_handle(self, label):
        _setup_crop(label)
        # Кружечок TL зміщений на 20% всередину. Позиція між кутом і кружечком,
        # що потрапляє в зону і кружечка, і хендла кадрування.
        # Візьмемо точку рівно на кружечку — вона в зоні обох (кружечок ближче до центру).
        tl_persp_widget = label._img_to_widget(label._persp_points[0])
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_persp_widget)
        label.mousePressEvent(ev)
        # Пріоритет у кружечка
        assert label._persp_drag_idx == 0
        assert label._crop_drag_idx == -1

    def test_no_drag_when_not_ready(self, label):
        # Рамка не готова — кружечки не активні
        label._crop_rect = [QPoint(10, 10), QPoint(90, 10), QPoint(90, 70), QPoint(10, 70)]
        label._crop_ready = False
        label._persp_points = [QPoint(20, 20), QPoint(80, 20), QPoint(80, 60), QPoint(20, 60)]
        tl_widget = label._img_to_widget(QPoint(20, 20))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_widget)
        label.mousePressEvent(ev)
        assert label._persp_drag_idx == -1

    def test_crop_handle_still_works(self, label):
        _setup_crop(label)
        # Клік точно на куті рамки (поза зоною кружечка)
        tl_crop_widget = label._img_to_widget(QPoint(10, 10))
        ev = _make_mouse_event(QEvent.Type.MouseButtonPress, tl_crop_widget)
        label.mousePressEvent(ev)
        assert label._crop_drag_idx == 0
        assert label._persp_drag_idx == -1


# ---------------------------------------------------------------------------
# mouseMoveEvent — незалежний drag кружечка
# ---------------------------------------------------------------------------

class TestMouseMovePersp:
    def test_drag_persp_independent(self, label):
        _setup_crop(label)
        label._persp_drag_idx = 0
        label._persp_detached[0] = True
        # Widget-позиція, що відповідає img_pt (30, 30)
        target_widget = label._img_to_widget(QPoint(30, 30))
        ev = _make_mouse_event(QEvent.Type.MouseMove, target_widget)
        label.mouseMoveEvent(ev)
        assert label._persp_points[0] == QPoint(30, 30)
        # Інші кружечки не змінились
        assert label._persp_points[1] != QPoint(30, 30)

    def test_drag_persp_does_not_move_crop(self, label):
        _setup_crop(label)
        crop_before = list(label._crop_rect)
        label._persp_drag_idx = 0
        label._persp_detached[0] = True
        target_widget = label._img_to_widget(QPoint(30, 30))
        ev = _make_mouse_event(QEvent.Type.MouseMove, target_widget)
        label.mouseMoveEvent(ev)
        # Рамка не змінилась
        assert label._crop_rect == crop_before


# ---------------------------------------------------------------------------
# mouseReleaseEvent
# ---------------------------------------------------------------------------

class TestMouseReleasePersp:
    def test_release_emits_and_resets(self, label):
        _setup_crop(label)
        label._persp_drag_idx = 0
        emitted = []
        label.persp_points_changed_hover.connect(lambda pts: emitted.append(pts))
        ev = _make_mouse_event(QEvent.Type.MouseButtonRelease, QPoint(0, 0))
        label.mouseReleaseEvent(ev)
        assert label._persp_drag_idx == -1
        assert len(emitted) == 1
        assert len(emitted[0]) == 4

    def test_release_without_drag_no_emit(self, label):
        _setup_crop(label)
        emitted = []
        label.persp_points_changed_hover.connect(lambda pts: emitted.append(pts))
        ev = _make_mouse_event(QEvent.Type.MouseButtonRelease, QPoint(0, 0))
        label.mouseReleaseEvent(ev)
        assert len(emitted) == 0


# ---------------------------------------------------------------------------
# _apply_crop_drag — синхронізація прив'язаних кружечків
# ---------------------------------------------------------------------------

class TestCropDragSyncPersp:
    def test_linked_follows_crop_drag(self, label):
        _setup_crop(label)
        # Всі кружечки прив'язані
        label._crop_drag_idx = 0
        label._apply_crop_drag(QPoint(20, 20))
        # Після drag TL(10,10)->(20,20): TR=(90,20) (ділить Y), BL=(20,70) (ділить X), BR=(90,70)
        # Новий центр: x=(20+90+90+20)/4=55, y=(20+20+70+70)/4=45
        # TL-кружечок: 20 + (55-20)*0.2 = 27, 20 + (45-20)*0.2 = 25
        assert label._persp_points[0] == QPoint(27, 25)

    def test_detached_does_not_follow(self, label):
        _setup_crop(label)
        # Відв'язуємо TL
        label._persp_detached[0] = True
        label._persp_points[0] = QPoint(5, 5)
        label._crop_drag_idx = 0
        label._apply_crop_drag(QPoint(20, 20))
        # Відв'язаний кружечок лишився на місці
        assert label._persp_points[0] == QPoint(5, 5)


class TestPerspectiveGuide:
    def test_linked_guide_matches_crop_corners(self, label):
        _setup_crop(label)

        guide = label._persp_guide_points()

        assert guide == [label._img_to_widget(point) for point in label._crop_rect]

    def test_detached_guide_uses_detached_point_only(self, label):
        _setup_crop(label)
        label._persp_detached[0] = True
        label._persp_points[0] = QPoint(30, 30)

        guide = label._persp_guide_points()

        assert guide[0] == label._img_to_widget(QPoint(30, 30))
        assert guide[1:] == [
            label._img_to_widget(point) for point in label._crop_rect[1:]
        ]

    def test_crop_state_uses_crop_corners_for_linked_points(self, label):
        _setup_crop(label)
        label._persp_detached[0] = True
        label._persp_points[0] = QPoint(30, 30)

        crop, perspective, detached = label.get_crop_state()

        assert perspective[0] == QPoint(30, 30)
        assert perspective[1:] == crop[1:]
        assert detached == [True, False, False, False]

    def test_dragging_perspective_point_does_not_expand_crop_bounds(self, label):
        _setup_crop(label)
        crop_before = list(label._crop_rect)
        label._persp_detached[0] = True
        label._persp_points[0] = QPoint(0, 0)
        # Perspective point is independent, crop bounds remain authoritative.
        assert label._crop_rect == crop_before


# ---------------------------------------------------------------------------
# leaveEvent — під час drag кружечка не скидає стан
# ---------------------------------------------------------------------------

class TestLeaveEventPersp:
    def test_leave_during_persp_drag_keeps_state(self, label):
        _setup_crop(label)
        label._persp_drag_idx = 0
        label.leaveEvent(None)
        assert label._persp_drag_idx == 0
