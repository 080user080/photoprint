"""Явна модель взаємовиключних режимів редагування MainWindow."""

from dataclasses import dataclass
from enum import Enum


class EditMode(str, Enum):
    IDLE = "idle"
    PERSPECTIVE = "perspective"
    DESKEW_PENDING = "deskew_pending"
    CROP = "crop"


@dataclass
class EditSession:
    """Єдине коротке джерело відповіді «який режим активний»."""

    mode: EditMode = EditMode.IDLE

    @property
    def active(self) -> bool:
        return self.mode is not EditMode.IDLE
