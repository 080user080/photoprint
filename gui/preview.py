"""
Прев'ю До/Після.
ImageLabel підтримує режим редагування 4 точок перспективи.
Точки завжди в межах видимого зображення.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore    import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui     import QPixmap, QImage, QPainter, QPen, QColor, QBrush
import numpy as np
import cv2

# Константи для ImageLabel
POINT_RADIUS = 9
POINT_HIT_RADIUS_MULTIPLIER = 2
POINT_HIT_TOLERANCE = 4
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

    _COLORS = [COLOR_TL, COLOR_TR, COLOR_BR, COLOR_BL]
    _LABELS = [LABEL_TL, LABEL_TR, LABEL_BR, LABEL_BL]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)
        # Залишаємо місце навколо для точок що на краю
        self.setContentsMargins(IMAGE_MARGIN, IMAGE_MARGIN, IMAGE_MARGIN, IMAGE_MARGIN)
        self._pixmap_orig: QPixmap | None = None
        self._img_w = 1
        self._img_h = 1
        self._points: list[QPoint] = []
        self._drag_idx: int = -1
        self._edit_mode: bool = False

    # --- Публічний API ---

    def set_image(self, image: np.ndarray):
        self._pixmap_orig = _np_to_pixmap(image)
        self._img_w = image.shape[1]
        self._img_h = image.shape[0]
        self._fit()

    def set_placeholder(self, text="Перетягніть зображення сюди"):
        self._pixmap_orig = None
        self._points = []
        self.clear()
        self.setText(text)
        self.setStyleSheet("color: #777777; font-size: 13px;")

    def set_edit_mode(self, enabled: bool, corners: list[QPoint] | None = None):
        self._edit_mode = enabled
        if enabled and corners:
            # Клампуємо точки щоб не виходили за межі зображення
            self._points = [self._clamp_to_image(p) for p in corners]
        elif not enabled:
            self._points = []
        self.update()

    def get_points(self) -> list[QPoint]:
        return list(self._points)

    # --- Qt events ---

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._edit_mode or len(self._points) != CORNER_COUNT:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

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

    def mousePressEvent(self, event):
        if not self._edit_mode:
            return
        pos = event.pos()
        for i, pt in enumerate(self._points):
            dp = self._img_to_widget(pt)
            if (pos - dp).manhattanLength() <= POINT_RADIUS * POINT_HIT_RADIUS_MULTIPLIER + POINT_HIT_TOLERANCE:
                self._drag_idx = i
                return
        self._drag_idx = -1

    def mouseMoveEvent(self, event):
        if not self._edit_mode or self._drag_idx < 0:
            return
        img_pt = self._widget_to_img(event.pos())
        if img_pt:
            self._points[self._drag_idx] = img_pt
            self.update()
            self.points_changed.emit(list(self._points))

    def mouseReleaseEvent(self, event):
        self._drag_idx = -1

    # --- Внутрішнє ---

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
        self.setStyleSheet("")

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

    def _clamp_to_image(self, pt: QPoint) -> QPoint:
        # Дозволяємо точки виходити за межі зображення на 20%
        margin_x = int(self._img_w * 0.2)
        margin_y = int(self._img_h * 0.2)
        x = max(-margin_x, min(pt.x(), self._img_w - 1 + margin_x))
        y = max(-margin_y, min(pt.y(), self._img_h - 1 + margin_y))
        return QPoint(x, y)


class PreviewPanel(QWidget):
    perspective_points_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT_SPACING)

        # До
        left = QVBoxLayout()
        lbl_b = QLabel("ДО")
        lbl_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_b.setStyleSheet("font-weight:bold; color:#333333; font-size:13px;")
        self._before = ImageLabel()
        self._before.set_placeholder()
        self._before.points_changed.connect(self.perspective_points_changed)
        left.addWidget(lbl_b)
        left.addWidget(self._before)

        # Після
        right = QVBoxLayout()
        self._lbl_after = QLabel("ПІСЛЯ")
        self._lbl_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_after.setStyleSheet("font-weight:bold; color:#333333; font-size:13px;")
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

    def set_autofix_applied(self, applied: bool):
        """Візуальний індикатор застосованої автокорекції."""
        if applied:
            self._lbl_after.setText("✓ ПІСЛЯ (Auto Fix)")
            self._lbl_after.setStyleSheet("font-weight:bold; color:#006600; font-size:13px;")
        else:
            self._lbl_after.setText("ПІСЛЯ")
            self._lbl_after.setStyleSheet("font-weight:bold; color:#333333; font-size:13px;")

    def clear(self):
        self._before.set_placeholder()
        self._after.set_placeholder("Тут з'явиться результат")
        self.set_autofix_applied(False)

    def enable_perspective_edit(self, corners: list[QPoint] | None = None):
        self._before.set_edit_mode(True, corners)

    def disable_perspective_edit(self):
        self._before.set_edit_mode(False)

    def get_perspective_points(self) -> list[QPoint]:
        return self._before.get_points()
