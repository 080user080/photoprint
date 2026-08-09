"""
Прев'ю До/Після.
ImageLabel підтримує режим редагування 4 точок перспективи.
Точки завжди в межах видимого зображення.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore    import Qt, QPoint, QRect, pyqtSignal, QTimer
from PyQt6.QtGui     import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor
import numpy as np
import cv2

# Константи для ImageLabel
POINT_RADIUS = 12
POINT_HIT_RADIUS_MULTIPLIER = 3
POINT_HIT_TOLERANCE = 8
MIN_IMAGE_SIZE = 280
IMAGE_MARGIN = 12
FIT_PADDING = 24  # IMAGE_MARGIN * 2

# Константи для малювання
LINE_WIDTH = 2
LINE_ALPHA = 200
SHADOW_ALPHA = 80
SHADOW_OFFSET = 2
CORNER_COUNT = 4

# Константи для кольорів точок
COLOR_TL = QColor(220, 50,  50)   # червоний
COLOR_TR = QColor(50,  180, 50)   # зелений
COLOR_BR = QColor(50,  50,  220)  # синій
COLOR_BL = QColor(220, 160, 0)    # жовтий

# Константи для міток
LABEL_TL = "TL"
LABEL_TR = "TR"
LABEL_BR = "BR"
LABEL_BL = "BL"

# Механізм завершення hover-сесії (TODO2 крок 2)
HOVER_HIDE_DELAY_MS = 50

# Константи для курсора-кадрування (TODO2 крок 2.3)
CROP_CURSOR_SIZE = 24
CROP_CURSOR_CORNER = 8
CROP_CURSOR_LINE_WIDTH = 2

# Константи для хендлів прямокутного кадрування (TODO2 крок 4)
CROP_HANDLE_ARM_LENGTH = 20   # довжина "плеча" кутової дужки у widget-пікселях
CROP_HANDLE_THICKNESS = 3     # товщина лінії дужки
CROP_HANDLE_HIT_TOLERANCE = 14  # радіус зони влучання кліком біля кута (widget-пікселі)
CROP_HANDLE_COLOR = QColor(255, 200, 0)  # насичений жовто-оранжевий
CROP_MIN_SIZE_PX = 40         # мінімальний розмір сторони прямокутника (координати зображення)

# Константи для хендлів перспективи-кружечків (TODO2 крок 5)
PERSP_HANDLE_INSET_RATIO = 0.20   # зміщення кружечка всередину від кута (частка діагоналі до центру)
PERSP_HANDLE_RADIUS = 9           # радіус кружечка у widget-пікселях
PERSP_HANDLE_HIT_TOLERANCE = 10   # зона влучання кліком (widget-пікселі)
PERSP_HANDLE_COLOR_LINKED = QColor(120, 200, 255)    # світло-блакитний — прив'язаний до рамки
PERSP_HANDLE_COLOR_DETACHED = QColor(255, 90, 200)   # яскраво-рожевий — відв'язаний
PERSP_HANDLE_FILL_ALPHA = 140     # прозорість заливки кружечка

# Константи для PreviewPanel
LAYOUT_SPACING = 8


def _np_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImageLabel(QLabel):
    """QLabel з підтримкою drag 4 точок перспективи."""

    points_changed = pyqtSignal(list)   # list[QPoint] у координатах зображення
    points_released = pyqtSignal(list)  # list[QPoint] — емітується лише в mouseReleaseEvent
    crop_session_requested = pyqtSignal()  # TODO2 крок 3.1: запит стартової рамки кадрування
    crop_rect_changed = pyqtSignal(list)   # TODO2 крок 4.4: list[QPoint] — під час drag хендла кадрування
    crop_rect_released = pyqtSignal(list)  # TODO2 крок 4.5: list[QPoint] — при відпусканні хендла кадрування
    persp_points_changed_hover = pyqtSignal(list)  # TODO2 крок 5.7: list[QPoint] — при відпусканні кружечка перспективи
    crop_session_committed = pyqtSignal()  # TODO2 крок 6.3: завершення зміненої hover-сесії

    _COLORS = [COLOR_TL, COLOR_TR, COLOR_BR, COLOR_BL]
    _LABELS = [LABEL_TL, LABEL_TR, LABEL_BR, LABEL_BL]
    _CROP_CURSOR: QCursor | None = None
    # TODO2 крок 4.4: таблиці сусідства для утримання прямокутної форми під час drag.
    # _ADJACENT_SHARE_Y[i] — індекс сусіда кута i, що ділить координату Y.
    # _ADJACENT_SHARE_X[i] — індекс сусіда кута i, що ділить координату X.
    _ADJACENT_SHARE_Y = {0: 1, 1: 0, 2: 3, 3: 2}
    _ADJACENT_SHARE_X = {0: 3, 1: 2, 2: 1, 3: 0}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)
        self._default_bg = "#E8E8E8"
        self.setStyleSheet(f"background-color: {self._default_bg}; border: 1px solid #CCCCCC; border-radius: 4px;")
        # Залишаємо місце навколо для точок що на краю
        self.setContentsMargins(IMAGE_MARGIN, IMAGE_MARGIN, IMAGE_MARGIN, IMAGE_MARGIN)
        self._pixmap_orig: QPixmap | None = None
        self._img_w = 1
        self._img_h = 1
        self._points: list[QPoint] = []
        self._drag_idx: int = -1
        self._edit_mode: bool = False
        # TODO2 крок 3: рамка кадрування (4 точки в координатах зображення)
        self._crop_rect: list[QPoint] = []
        self._crop_rect_requested_for_current_image: bool = False
        # TODO2 крок 4: стан хендлів кадрування
        self._crop_ready: bool = False   # рамка отримана і готова до редагування
        self._crop_drag_idx: int = -1    # індекс кута, який зараз тягнуть (-1 — нічого)
        self._crop_rect_drag_snapshot: list[QPoint] | None = None
        # TODO2 крок 5: стан хендлів перспективи-кружечків
        self._persp_points: list[QPoint] = []          # 4 точки кружечків (порядок [TL, TR, BR, BL])
        self._persp_detached: list[bool] = [False, False, False, False]  # "відв'язаний від рамки" для кожного
        self._persp_drag_idx: int = -1                 # індекс кружечка, який зараз тягнуть (-1 — нічого)
        self._persp_point_drag_snapshot: QPoint | None = None
        self._persp_detached_drag_snapshot: bool | None = None
        self._crop_session_dirty: bool = False         # чи був відпущений хоча б один хендл
        # Hover-стан: текстовий overlay прибрано, але сесія потрібна для
        # миттєвої рамки, курсора та commit при виході.
        self._hover_enabled: bool = True
        self._hover_hide_timer = QTimer(self)
        self._hover_hide_timer.setSingleShot(True)
        self._hover_hide_timer.setInterval(HOVER_HIDE_DELAY_MS)
        self._hover_hide_timer.timeout.connect(self._hide_hover_overlay)

    # --- Публічний API ---

    def set_hover_enabled(self, enabled: bool) -> None:
        """TODO2 крок 2: вмикає/вимикає hover-оверлей (напр. під час активної перспективи).

        Якщо вимкнено — негайно ховає оверлей, зупиняє обидва таймери
        і повертає стандартний курсор.
        """
        self._hover_enabled = enabled
        if not enabled:
            self._hover_hide_timer.stop()
            self.unsetCursor()
            self.update()

    def set_image(self, image: np.ndarray):
        self._pixmap_orig = _np_to_pixmap(image)
        self._img_w = image.shape[1]
        self._img_h = image.shape[0]
        self._disable_hover()
        self._fit()

    def set_placeholder(self, text="Перетягніть зображення сюди"):
        self._pixmap_orig = None
        self._points = []
        self.clear()
        self.setText(text)
        text_color = "#777777"
        self.setStyleSheet(f"background-color: {self._default_bg}; border: 1px solid #CCCCCC; border-radius: 4px; color: {text_color}; font-size: 13px;")
        self._disable_hover()

    def set_edit_mode(self, enabled: bool, corners: list[QPoint] | None = None):
        self._edit_mode = enabled
        if enabled and corners:
            # Клампуємо точки щоб не виходили за межі зображення
            self._points = [self._clamp_to_image(p) for p in corners]
        elif not enabled:
            self._points = []
        # Під час активної перспективи hover-оверлей не показуємо
        if enabled:
            self.set_hover_enabled(False)
        else:
            self.unsetCursor()
        self.update()

    def get_points(self) -> list[QPoint]:
        return list(self._points)

    def set_crop_rect(self, corners: list[QPoint]) -> None:
        """TODO2 крок 3.6/4.0/5.3: зберігає рамку кадрування (4 точки в координатах зображення).

        Отримання рамки означає готовність до редагування — виставляємо
        `_crop_ready = True`, щоб хендли кадрування стали активними.
        TODO2 крок 5.3: нова hover-сесія = новий незалежний знімок —
        скидаємо `_persp_detached` і перераховуємо кружечки як похідні від рамки.
        """
        self._crop_rect = [self._clamp_crop_point(point) for point in corners]
        self._crop_ready = True
        # TODO2 крок 5.3: скидання стану перспективи на нову hover-сесію
        self._persp_detached = [False, False, False, False]
        self._persp_drag_idx = -1
        self._sync_linked_persp_points()
        self.update()

    def set_crop_ready(self, ready: bool) -> None:
        """TODO2 крок 4.0: вмикає/вимикає готовність хендлів кадрування.

        Викликається з MainWindow зі значенням False на старті фонової
        детекції кутів (поки рамка ще не готова — хендли не показуємо
        і не дозволяємо drag). True зазвичай виставляється автоматично
        через set_crop_rect().
        """
        self._crop_ready = ready
        if not ready:
            self._crop_drag_idx = -1
        self.update()

    # --- Qt events ---

    def resizeEvent(self, event):
        if self._crop_drag_idx >= 0 or self._persp_drag_idx >= 0:
            self._abort_crop_session_due_to_resize()
        super().resizeEvent(event)
        self._fit()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # TODO2 крок 4.2: хендли прямокутного кадрування (кутові дужки)
        if self._crop_ready and len(self._crop_rect) == CORNER_COUNT:
            self._draw_crop_handles(painter)
        # TODO2 крок 5.4: хендли перспективи-кружечки
        if self._crop_ready and len(self._persp_points) == CORNER_COUNT:
            self._draw_persp_handles(painter)
        if self._edit_mode and len(self._points) == CORNER_COUNT:
            # Лінії між точками
            pen = QPen(QColor(255, 255, 255, LINE_ALPHA), LINE_WIDTH, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for i in range(CORNER_COUNT):
                d1 = self._img_to_widget(self._points[i])
                d2 = self._img_to_widget(self._points[(i + 1) % CORNER_COUNT])
                painter.drawLine(d1, d2)

            # Точки
            for i, pt in enumerate(self._points):
                dp = self._img_to_widget(pt)
                color = self._COLORS[i]
                # Тінь
                painter.setBrush(QBrush(QColor(0, 0, 0, SHADOW_ALPHA)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(dp.x() - POINT_RADIUS + SHADOW_OFFSET,
                                    dp.y() - POINT_RADIUS + SHADOW_OFFSET,
                                    POINT_RADIUS * 2, POINT_RADIUS * 2)
                # Кружок
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255), LINE_WIDTH))
                painter.drawEllipse(dp.x() - POINT_RADIUS,
                                    dp.y() - POINT_RADIUS,
                                    POINT_RADIUS * 2, POINT_RADIUS * 2)
                # Мітка
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(dp.x() - 8, dp.y() + 4, self._LABELS[i])

    def enterEvent(self, event):
        super().enterEvent(event)
        self._maybe_schedule_hover(event.position().toPoint())

    def leaveEvent(self, event):
        super().leaveEvent(event)
        # TODO2 крок 4.6/5.9: під час активного drag хендла кадрування або
        # кружечка перспективи не скидаємо курсор і не ховаємо оверлей —
        # інакше курсор "мигатиме" при виході за межу зображення/віджета.
        if self._crop_drag_idx >= 0 or self._persp_drag_idx >= 0:
            return
        # Приховування — мінімальна затримка (HOVER_HIDE_DELAY_MS)
        self.unsetCursor()
        if self._crop_session_dirty:
            self._hover_hide_timer.start()

    def mouseMoveEvent(self, event):
        # TODO2 крок 5.6: перетягування кружечка перспективи (пріоритет над рамкою)
        if self._persp_drag_idx >= 0:
            img_pt = self._widget_to_img(event.pos())
            if img_pt is not None:
                self._persp_points[self._persp_drag_idx] = img_pt
                self._expand_crop_to_include_persp_points()
                self.update()
                # Навмисно НЕ викликаємо _sync_linked_persp_points() тут:
                # кружечок уже _persp_detached[i] = True, і синхронізація
                # його б не чіпала, але явно уникаємо виклику, щоб майбутні
                # правки не додали його помилково (TODO2 крок 5.6).
            return
        # TODO2 крок 4.4: перетягування хендла кадрування (пріоритет над hover)
        if self._crop_drag_idx >= 0:
            # Рамка кадрування не може виходити за межі видимого зображення.
            img_pt = self._widget_to_crop_img(event.pos())
            if img_pt is not None:
                self._apply_crop_drag(img_pt)
            return
        if self._edit_mode and self._drag_idx >= 0:
            img_pt = self._widget_to_img(event.pos())
            if img_pt is not None:
                self._points[self._drag_idx] = img_pt
                self.update()
                self.points_changed.emit(list(self._points))
            return
        # Hover-детекція (TODO2 крок 2)
        self._maybe_schedule_hover(event.pos())

    def mousePressEvent(self, event):
        if self._edit_mode:
            pos = event.pos()
            for i, pt in enumerate(self._points):
                dp = self._img_to_widget(pt)
                if (pos - dp).manhattanLength() <= POINT_RADIUS * POINT_HIT_RADIUS_MULTIPLIER + POINT_HIT_TOLERANCE:
                    self._drag_idx = i
                    return
            self._drag_idx = -1
            return
        # Швидкий клік під час появи підказки спочатку активує crop-сесію.
        # Підказка не є окремим віджетом і не повинна блокувати drag.
        if not self._crop_ready:
            self._maybe_schedule_hover(event.pos())
        # TODO2 крок 5.5: hit-test кружечків перспективи (пріоритет над хендлами кадрування)
        if self._crop_ready and len(self._persp_points) == CORNER_COUNT:
            pos = event.pos()
            for i, pt in enumerate(self._persp_points):
                dp = self._img_to_widget(pt)
                if (pos - dp).manhattanLength() <= PERSP_HANDLE_HIT_TOLERANCE:
                    self._persp_point_drag_snapshot = QPoint(self._persp_points[i])
                    self._persp_detached_drag_snapshot = self._persp_detached[i]
                    self._persp_drag_idx = i
                    # Розрив зв'язку з рамкою — кружечок стає незалежним
                    self._persp_detached[i] = True
                    self.update()
                    return
        # TODO2 крок 4.3: hit-test хендлів кадрування (тільки коли рамка готова)
        if self._crop_ready and len(self._crop_rect) == CORNER_COUNT:
            pos = event.pos()
            for i, pt in enumerate(self._crop_rect):
                dp = self._img_to_widget(pt)
                if (pos - dp).manhattanLength() <= CROP_HANDLE_HIT_TOLERANCE:
                    self._crop_rect_drag_snapshot = list(self._crop_rect)
                    self._crop_drag_idx = i
                    self.update()
                    return
            self._crop_drag_idx = -1

    def mouseReleaseEvent(self, event):
        self._drag_idx = -1
        # TODO2 крок 5.7: завершення drag кружечка перспективи
        if self._persp_drag_idx >= 0:
            self._persp_drag_idx = -1
            self._persp_point_drag_snapshot = None
            self._persp_detached_drag_snapshot = None
            self._crop_session_dirty = True
            self.persp_points_changed_hover.emit(list(self._persp_points))
        # TODO2 крок 4.5: завершення drag хендла кадрування
        if self._crop_drag_idx >= 0:
            self._crop_drag_idx = -1
            self._crop_rect_drag_snapshot = None
            self._crop_session_dirty = True
            self.crop_rect_released.emit(list(self._crop_rect))
        if self._edit_mode and len(self._points) == CORNER_COUNT:
            self.points_released.emit(list(self._points))

    # --- Внутрішнє ---

    def _maybe_schedule_hover(self, event_pos: QPoint | None = None):
        """Активує crop-рамку й курсор, якщо курсор над зображенням."""
        if not self._hover_enabled:
            return
        rect = self._img_rect()
        if rect is None:
            return
        inside_pos = event_pos
        if inside_pos is None:
            inside_pos = self.mapFromGlobal(self.cursor().pos())
        inside = rect.contains(inside_pos)
        if not inside:
            # Курсор поза зображенням — скасовуємо запланований показ
            self.unsetCursor()
            return
        if self._edit_mode:
            return
        # Рамка й crop-курсор активуються одразу при вході в область «ДО».
        if not self._crop_rect_requested_for_current_image:
            self._crop_rect_requested_for_current_image = True
            self.crop_session_requested.emit()
        # Курсор-кадрування — миттєво, без затримки
        self.setCursor(self._get_crop_cursor())
    def _hide_hover_overlay(self):
        """Завершує hover-сесію після виходу курсора із зображення."""
        if self._crop_ready and self._crop_session_dirty:
            self.crop_session_committed.emit()
            self._crop_session_dirty = False
        self.update()

    def _disable_hover(self):
        """Скидає hover-стан (при зміні зображення/placeholder)."""
        self._hover_hide_timer.stop()
        self.unsetCursor()
        # TODO2 крок 3.1: наступне наведення знову має право запросити рамку
        self._crop_rect_requested_for_current_image = False
        self._crop_rect = []
        # TODO2 крок 4: скидаємо стан хендлів кадрування
        self._crop_ready = False
        self._crop_drag_idx = -1
        self._crop_rect_drag_snapshot = None
        # TODO2 крок 5: скидаємо стан хендлів перспективи-кружечків
        self._persp_points = []
        self._persp_detached = [False, False, False, False]
        self._persp_drag_idx = -1
        self._persp_point_drag_snapshot = None
        self._persp_detached_drag_snapshot = None
        self._crop_session_dirty = False

    def _abort_crop_session_due_to_resize(self) -> None:
        """Перериває hover-crop drag при зміні розміру віджета.

        Метод навмисно не викликає `_hide_hover_overlay()`: resize не має
        емітити `crop_session_committed` і не може змінити `_base` у MainWindow.
        """
        crop_idx = self._crop_drag_idx
        persp_idx = self._persp_drag_idx
        if crop_idx >= 0 and self._crop_rect_drag_snapshot is not None:
            self._crop_rect = list(self._crop_rect_drag_snapshot)
        if (
            persp_idx >= 0
            and self._persp_point_drag_snapshot is not None
            and len(self._persp_points) == CORNER_COUNT
        ):
            self._persp_points[persp_idx] = QPoint(self._persp_point_drag_snapshot)
            if self._persp_detached_drag_snapshot is not None:
                self._persp_detached[persp_idx] = self._persp_detached_drag_snapshot

        self._crop_drag_idx = -1
        self._persp_drag_idx = -1
        self._crop_rect_drag_snapshot = None
        self._persp_point_drag_snapshot = None
        self._persp_detached_drag_snapshot = None

        # Варіант A: повністю відкидаємо незавершену hover-crop сесію.
        self._crop_rect = []
        self._persp_points = []
        self._persp_detached = [False, False, False, False]
        self._crop_ready = False
        self._crop_session_dirty = False
        self._crop_rect_requested_for_current_image = False

        self._hover_hide_timer.stop()
        self.unsetCursor()
        self.update()

    def get_crop_state(self) -> tuple[list[QPoint], list[QPoint], list[bool]]:
        """Повертає фінальний стан рамки й кружечків у preview-координатах."""
        return (
            list(self._crop_rect),
            list(self._persp_points),
            list(self._persp_detached),
        )

    def reset_crop_session(self) -> None:
        """Завершує hover-сесію без повторного сигналу коміту."""
        self._disable_hover()

    def _compute_linked_persp_point(self, corner_idx: int) -> QPoint:
        """TODO2 крок 5.2: обчислює позицію «прив'язаного» кружечка.

        Кружечок зміщений від відповідного кута `_crop_rect[corner_idx]`
        на `PERSP_HANDLE_INSET_RATIO` у напрямку центру прямокутника.
        Координати — в системі зображення (як і сам `_crop_rect`).
        """
        if len(self._crop_rect) != CORNER_COUNT:
            return QPoint(0, 0)
        corner = self._crop_rect[corner_idx]
        cx = sum(p.x() for p in self._crop_rect) // CORNER_COUNT
        cy = sum(p.y() for p in self._crop_rect) // CORNER_COUNT
        x = corner.x() + int((cx - corner.x()) * PERSP_HANDLE_INSET_RATIO)
        y = corner.y() + int((cy - corner.y()) * PERSP_HANDLE_INSET_RATIO)
        return QPoint(x, y)

    def _sync_linked_persp_points(self) -> None:
        """TODO2 крок 5.2: оновлює позиції «прив'язаних» кружечків.

        Для кожного i, якщо `not self._persp_detached[i]` — перераховує
        `_persp_points[i]` як похідну від відповідного кута рамки.
        «Відв'язані» кружечки цей метод не чіпає.
        """
        if len(self._crop_rect) != CORNER_COUNT:
            return
        if len(self._persp_points) != CORNER_COUNT:
            self._persp_points = [QPoint(0, 0)] * CORNER_COUNT
        for i in range(CORNER_COUNT):
            if not self._persp_detached[i]:
                self._persp_points[i] = self._compute_linked_persp_point(i)

    def _expand_crop_to_include_persp_points(self) -> None:
        """Розширює crop-рамку, якщо перспектива вийшла за її межі.

        Розширення обмежене межами зображення, тому подальше кадрування не
        відріже частину зображення, яку користувач включив у перспективу.
        Відв'язані кружечки не прив'язуються назад до рамки.
        """
        if len(self._crop_rect) != CORNER_COUNT or len(self._persp_points) != CORNER_COUNT:
            return
        crop_xs = [point.x() for point in self._crop_rect]
        crop_ys = [point.y() for point in self._crop_rect]
        persp_xs = [point.x() for point in self._persp_points]
        persp_ys = [point.y() for point in self._persp_points]
        x_min = max(0, min(crop_xs + persp_xs))
        y_min = max(0, min(crop_ys + persp_ys))
        x_max = min(self._img_w - 1, max(crop_xs + persp_xs))
        y_max = min(self._img_h - 1, max(crop_ys + persp_ys))

        current_bounds = (min(crop_xs), max(crop_xs), min(crop_ys), max(crop_ys))
        new_bounds = (x_min, x_max, y_min, y_max)
        if new_bounds == current_bounds:
            return

        self._crop_rect = [
            QPoint(x_min, y_min), QPoint(x_max, y_min),
            QPoint(x_max, y_max), QPoint(x_min, y_max),
        ]
        self._sync_linked_persp_points()
        self.crop_rect_changed.emit(list(self._crop_rect))

    @classmethod
    def _get_crop_cursor(cls) -> QCursor:
        """TODO2 крок 2.3: повертає закешований курсор-кадрування.

        Малює 24×24 px курсор з двома «кутиками» (як у Photoshop crop tool).
        Створюється один раз при першому виклику, далі повертається з кешу.
        """
        if cls._CROP_CURSOR is not None:
            return cls._CROP_CURSOR
        pm = QPixmap(CROP_CURSOR_SIZE, CROP_CURSOR_SIZE)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(255, 255, 255), CROP_CURSOR_LINE_WIDTH + 1)  # біла тінь
        painter.setPen(pen)
        # Верхній лівий кутик
        painter.drawLine(0, CROP_CURSOR_CORNER, 0, 0)
        painter.drawLine(0, 0, CROP_CURSOR_CORNER, 0)
        # Нижній правий кутик
        painter.drawLine(CROP_CURSOR_SIZE - 1, CROP_CURSOR_SIZE - 1 - CROP_CURSOR_CORNER,
                         CROP_CURSOR_SIZE - 1, CROP_CURSOR_SIZE - 1)
        painter.drawLine(CROP_CURSOR_SIZE - 1, CROP_CURSOR_SIZE - 1,
                         CROP_CURSOR_SIZE - 1 - CROP_CURSOR_CORNER, CROP_CURSOR_SIZE - 1)
        pen = QPen(QColor(0, 0, 0), CROP_CURSOR_LINE_WIDTH)  # чорний основний
        painter.setPen(pen)
        # Верхній лівий кутик
        painter.drawLine(0, CROP_CURSOR_CORNER, 0, 0)
        painter.drawLine(0, 0, CROP_CURSOR_CORNER, 0)
        # Нижній правий кутик
        painter.drawLine(CROP_CURSOR_SIZE - 1, CROP_CURSOR_SIZE - 1 - CROP_CURSOR_CORNER,
                         CROP_CURSOR_SIZE - 1, CROP_CURSOR_SIZE - 1)
        painter.drawLine(CROP_CURSOR_SIZE - 1, CROP_CURSOR_SIZE - 1,
                         CROP_CURSOR_SIZE - 1 - CROP_CURSOR_CORNER, CROP_CURSOR_SIZE - 1)
        painter.end()
        cls._CROP_CURSOR = QCursor(pm, 0, 0)
        return cls._CROP_CURSOR

    def _draw_crop_handles(self, painter: QPainter) -> None:
        """TODO2 крок 4.2: малює прямокутник кадрування та 4 кутові дужки-хендли.

        Дужки у формі букви «Г» виходять з кожного кута всередину прямокутника
        (напрямок обчислюється динамічно за напрямком до центру фігури).
        """
        # Пунктирна рамка по 4 сторонах
        pen = QPen(QColor(255, 255, 255, LINE_ALPHA), LINE_WIDTH, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(CORNER_COUNT):
            d1 = self._img_to_widget(self._crop_rect[i])
            d2 = self._img_to_widget(self._crop_rect[(i + 1) % CORNER_COUNT])
            painter.drawLine(d1, d2)

        # Центр прямокутника (у widget-координатах) — для напрямку дужок
        pts = [self._img_to_widget(p) for p in self._crop_rect]
        cx = sum(p.x() for p in pts) // CORNER_COUNT
        cy = sum(p.y() for p in pts) // CORNER_COUNT

        # Кутові дужки-хендли
        for dp in pts:
            # Напрямок до центру (нормалізований до ±1 по кожній осі)
            dx = 1 if cx > dp.x() else -1
            dy = 1 if cy > dp.y() else -1
            # Тінь
            painter.setPen(QPen(QColor(0, 0, 0, SHADOW_ALPHA), CROP_HANDLE_THICKNESS))
            painter.drawLine(dp.x() + SHADOW_OFFSET, dp.y() + SHADOW_OFFSET,
                             dp.x() + dx * CROP_HANDLE_ARM_LENGTH + SHADOW_OFFSET,
                             dp.y() + SHADOW_OFFSET)
            painter.drawLine(dp.x() + SHADOW_OFFSET, dp.y() + SHADOW_OFFSET,
                             dp.x() + SHADOW_OFFSET,
                             dp.y() + dy * CROP_HANDLE_ARM_LENGTH + SHADOW_OFFSET)
            # Основний колір
            painter.setPen(QPen(CROP_HANDLE_COLOR, CROP_HANDLE_THICKNESS))
            painter.drawLine(dp.x(), dp.y(),
                             dp.x() + dx * CROP_HANDLE_ARM_LENGTH, dp.y())
            painter.drawLine(dp.x(), dp.y(),
                             dp.x(), dp.y() + dy * CROP_HANDLE_ARM_LENGTH)

    def _draw_persp_handles(self, painter: QPainter) -> None:
        """TODO2 крок 5.4: малює 4 кружечки-хендли перспективи.

        Колір кружечка залежить від прапорця `_persp_detached[i]`:
        LINKED (світло-блакитний) — ще слідує за рамкою,
        DETACHED (яскраво-рожевий) — вже незалежний.
        Без міток TL/TR/BR/BL; направляюча використовує лінію crop-рамки
        для прив'язаних точок.
        """
        # Для прив'язаних кружечків направляюча проходить точно по лінії
        # рамки кадрування, хоча сам центр кружечка залишається трохи
        # всередині, щоб не перекриватися з кутовою дужкою crop.
        guide_points = self._persp_guide_points()
        guide_pen = QPen(
            QColor(PERSP_HANDLE_COLOR_LINKED.red(),
                   PERSP_HANDLE_COLOR_LINKED.green(),
                   PERSP_HANDLE_COLOR_LINKED.blue(), LINE_ALPHA),
            LINE_WIDTH,
            Qt.PenStyle.DashLine,
        )
        painter.setPen(guide_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(CORNER_COUNT):
            painter.drawLine(
                guide_points[i], guide_points[(i + 1) % CORNER_COUNT]
            )

        for i, pt in enumerate(self._persp_points):
            dp = self._img_to_widget(pt)
            color = PERSP_HANDLE_COLOR_DETACHED if self._persp_detached[i] else PERSP_HANDLE_COLOR_LINKED
            # Тінь
            painter.setBrush(QBrush(QColor(0, 0, 0, SHADOW_ALPHA)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(dp.x() - PERSP_HANDLE_RADIUS + SHADOW_OFFSET,
                                dp.y() - PERSP_HANDLE_RADIUS + SHADOW_OFFSET,
                                PERSP_HANDLE_RADIUS * 2, PERSP_HANDLE_RADIUS * 2)
            # Кружечок: напівпрозора заливка + білий контур
            fill = QColor(color)
            fill.setAlpha(PERSP_HANDLE_FILL_ALPHA)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor(255, 255, 255), LINE_WIDTH))
            painter.drawEllipse(dp.x() - PERSP_HANDLE_RADIUS,
                                dp.y() - PERSP_HANDLE_RADIUS,
                                PERSP_HANDLE_RADIUS * 2, PERSP_HANDLE_RADIUS * 2)

    def _persp_guide_points(self) -> list[QPoint]:
        """Повертає widget-точки перспективної направляючої.

        Прив'язані точки малюються по кутах crop-рамки, а відв'язані — по
        власних координатах. Так лінія збігається з рамкою до моменту, коли
        користувач свідомо від'єднав конкретний перспективний кружечок.
        """
        if len(self._crop_rect) != CORNER_COUNT:
            return [self._img_to_widget(point) for point in self._persp_points]
        return [
            self._img_to_widget(
                self._crop_rect[i] if not self._persp_detached[i]
                else self._persp_points[i]
            )
            for i in range(CORNER_COUNT)
        ]

    def _apply_crop_drag(self, img_pt: QPoint) -> None:
        """TODO2 крок 4.4: оновлює _crop_rect, зберігаючи прямокутну форму.

        Тягнутий кут (self._crop_drag_idx) стає img_pt; сусідні кути, що
        ділять з ним координату X/Y, оновлюються відповідно; протилежний
        кут лишається нерухомим «якорем». Застосовується мінімальний розмір.
        """
        idx = self._crop_drag_idx
        if idx < 0 or len(self._crop_rect) != CORNER_COUNT:
            return
        rect = list(self._crop_rect)
        # Поточні межі прямокутника
        xs = [p.x() for p in rect]
        ys = [p.y() for p in rect]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Затискаємо рух по осі, якщо він порушив би мінімальний розмір.
        # Визначаємо, чи тягнутий кут є "лівим/правим" і "верхнім/нижнім".
        is_left = rect[idx].x() == x_min
        is_top = rect[idx].y() == y_min

        new_x = img_pt.x()
        new_y = img_pt.y()
        # Затискаємо рух по осі, щоб прямокутник не схлопнувся нижче мінімуму.
        # Для кута, що є "лівим" (x == x_min), обмежуємо зверху (x_max - min);
        # для "правого" — знизу (x_min + min). Аналогічно по Y.
        if is_left:
            new_x = min(new_x, x_max - CROP_MIN_SIZE_PX)
        else:
            new_x = max(new_x, x_min + CROP_MIN_SIZE_PX)
        if is_top:
            new_y = min(new_y, y_max - CROP_MIN_SIZE_PX)
        else:
            new_y = max(new_y, y_min + CROP_MIN_SIZE_PX)

        # Оновлюємо тягнутий кут
        rect[idx] = QPoint(new_x, new_y)
        # Сусід, що ділить Y
        share_y = self._ADJACENT_SHARE_Y[idx]
        rect[share_y] = QPoint(rect[share_y].x(), new_y)
        # Сусід, що ділить X
        share_x = self._ADJACENT_SHARE_X[idx]
        rect[share_x] = QPoint(new_x, rect[share_x].y())

        self._crop_rect = rect
        # TODO2 крок 5.8: «прив'язані» кружечки слідують за рухом рамки в реальному часі
        self._sync_linked_persp_points()
        self.update()
        self.crop_rect_changed.emit(list(self._crop_rect))

    def _fit(self):
        if self._pixmap_orig is None:
            return
        avail = self.size()
        scaled = self._pixmap_orig.scaled(
            avail.width() - FIT_PADDING, avail.height() - FIT_PADDING,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
        # Залишаємо фон сірим, прибираємо тільки текстову стилізацію
        self.setStyleSheet(f"background-color: {self._default_bg}; border: 1px solid #CCCCCC; border-radius: 4px;")

    def _img_rect(self) -> QRect | None:
        if not self.pixmap() or self.pixmap().isNull():
            return None
        pw = self.pixmap().width()
        ph = self.pixmap().height()
        x = (self.width()  - pw) // 2
        y = (self.height() - ph) // 2
        return QRect(x, y, pw, ph)

    def _img_to_widget(self, img_pt: QPoint) -> QPoint:
        rect = self._img_rect()
        if rect is None:
            return img_pt
        sx = rect.width()  / max(self._img_w, 1)
        sy = rect.height() / max(self._img_h, 1)
        return QPoint(rect.x() + int(img_pt.x() * sx),
                      rect.y() + int(img_pt.y() * sy))

    def _widget_to_img(self, pos: QPoint) -> QPoint | None:
        rect = self._img_rect()
        if rect is None:
            return None
        sx = max(self._img_w, 1) / max(rect.width(),  1)
        sy = max(self._img_h, 1) / max(rect.height(), 1)
        ix = int((pos.x() - rect.x()) * sx)
        iy = int((pos.y() - rect.y()) * sy)
        return self._clamp_to_image(QPoint(ix, iy))

    def _widget_to_crop_img(self, pos: QPoint) -> QPoint | None:
        """Перетворює координати віджета в координати crop без виходу назовні."""
        rect = self._img_rect()
        if rect is None:
            return None
        sx = max(self._img_w, 1) / max(rect.width(), 1)
        sy = max(self._img_h, 1) / max(rect.height(), 1)
        ix = int((pos.x() - rect.x()) * sx)
        iy = int((pos.y() - rect.y()) * sy)
        return QPoint(
            max(0, min(ix, self._img_w - 1)),
            max(0, min(iy, self._img_h - 1)),
        )

    def _clamp_crop_point(self, pt: QPoint) -> QPoint:
        """Keep the rectangular crop frame inside the actual image bounds."""
        return QPoint(
            max(0, min(pt.x(), self._img_w - 1)),
            max(0, min(pt.y(), self._img_h - 1)),
        )

    def _clamp_to_image(self, pt: QPoint) -> QPoint:
        # Дозволяємо точки виходити за межі зображення на 20%
        margin_x = int(self._img_w * 0.2)
        margin_y = int(self._img_h * 0.2)
        x = max(-margin_x, min(pt.x(), self._img_w - 1 + margin_x))
        y = max(-margin_y, min(pt.y(), self._img_h - 1 + margin_y))
        return QPoint(x, y)


class PreviewPanel(QWidget):
    perspective_points_changed = pyqtSignal(list)
    perspective_points_released = pyqtSignal(list)
    crop_session_requested = pyqtSignal()  # TODO2 крок 3.4: проксі-сигнал запиту рамки кадрування
    crop_session_committed = pyqtSignal()  # TODO2 крок 6.4: проксі-сигнал коміту hover-сесії
    crop_rect_released = pyqtSignal(list)
    persp_points_changed_hover = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._default_bg = "#E8E8E8"
        self._text_color = "#333333"
        self._autofix_mode: str | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT_SPACING)

        # До
        left = QVBoxLayout()
        self._lbl_before = QLabel("ДО")
        self._lbl_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_before.setStyleSheet("font-weight:bold; color:#333333; font-size:13px;")
        self._before = ImageLabel()
        self._before.set_placeholder()
        self._before.points_changed.connect(self.perspective_points_changed)
        self._before.points_released.connect(self.perspective_points_released)
        self._before.crop_session_requested.connect(self.crop_session_requested)
        self._before.crop_session_committed.connect(self.crop_session_committed)
        self._before.crop_rect_released.connect(self.crop_rect_released)
        self._before.persp_points_changed_hover.connect(self.persp_points_changed_hover)
        left.addWidget(self._lbl_before)
        left.addWidget(self._before)

        # Після
        right = QVBoxLayout()
        self._lbl_after = QLabel("ПІСЛЯ")
        self._lbl_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_after.setStyleSheet("font-weight:bold; color:_preview_text; font-size:13px;")
        self._after = ImageLabel()
        self._after.set_placeholder("Тут з'явиться результат")
        right.addWidget(self._lbl_after)
        right.addWidget(self._after)

        layout.addLayout(left)
        layout.addLayout(right)

    def set_before(self, image: np.ndarray):
        self._before.set_image(image)

    def set_after(self, image: np.ndarray):
        self._after.set_image(image)

    def set_colors(self, bg_color: str, text_color: str):
        """Встановити кольори фону та тексту для прев'ю."""
        self._default_bg = bg_color
        self._text_color = text_color
        if hasattr(self, '_before'):
            self._before._default_bg = bg_color
        if hasattr(self, '_after'):
            self._after._default_bg = bg_color
        self._lbl_before.setStyleSheet(f"font-weight:bold; color:{text_color}; font-size:13px;")
        self._lbl_after.setStyleSheet(f"font-weight:bold; color:{text_color}; font-size:13px;")
        # Оновлюємо фон ImageLabel
        for attr in ('_before', '_after'):
            w = getattr(self, attr)
            w.setStyleSheet(f"background-color: {bg_color}; border: 1px solid #CCCCCC; border-radius: 4px;")

    def set_autofix_applied(self, mode: str | None):
        """Візуальний індикатор застосованої автокорекції.

        Параметри:
            mode: "auto_fix" / "full_auto" / None (скидання)
        """
        self._autofix_mode = mode
        if mode == "auto_fix":
            self._lbl_after.setText("✓ ПІСЛЯ (Auto Fix)")
            self._lbl_after.setStyleSheet("font-weight:bold; color:#006600; font-size:13px;")
        elif mode == "full_auto":
            self._lbl_after.setText("✓ ПІСЛЯ (Full Auto)")
            self._lbl_after.setStyleSheet("font-weight:bold; color:#006600; font-size:13px;")
        else:
            self._lbl_after.setText("ПІСЛЯ")
            self._lbl_after.setStyleSheet(f"font-weight:bold; color:{self._text_color}; font-size:13px;")

    def clear(self):
        self._before.set_placeholder()
        self._after.set_placeholder("Тут з'явиться результат")
        self.set_autofix_applied(None)

    def get_autofix_applied(self) -> str | None:
        """Повертає поточний режим індикатора («auto_fix»/«full_auto»/None)."""
        return self._autofix_mode

    def enable_perspective_edit(self, corners: list[QPoint] | None = None):
        self._before.set_edit_mode(True, corners)

    def disable_perspective_edit(self):
        self._before.set_edit_mode(False)
        # Після виходу з перспективи — hover-оверлей знову доступний (TODO2 крок 2)
        self._before.set_hover_enabled(True)

    def set_hover_enabled(self, enabled: bool) -> None:
        """TODO2 крок 2: вмикає/вимикає hover-оверлей на панелі «ДО»."""
        self._before.set_hover_enabled(enabled)

    def get_perspective_points(self) -> list[QPoint]:
        return self._before.get_points()

    def get_crop_state(self) -> tuple[list[QPoint], list[QPoint], list[bool]]:
        """Повертає стан кадрування з ImageLabel у preview-координатах."""
        return self._before.get_crop_state()

    def reset_crop_session(self) -> None:
        """Скидає hover-сесію панелі після фінального коміту."""
        self._before.reset_crop_session()
