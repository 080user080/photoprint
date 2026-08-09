"""Перетворення координат між повнорозмірним зображенням і preview."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def image_to_preview_points(
    points: Sequence[Sequence[float]],
    source_shape: tuple[int, int],
    preview_shape: tuple[int, int],
) -> np.ndarray:
    """Масштабує точки з координат source у координати preview."""
    src_h, src_w = source_shape
    preview_h, preview_w = preview_shape
    scale_x = preview_w / max(src_w, 1)
    scale_y = preview_h / max(src_h, 1)
    result = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    result[:, 0] *= scale_x
    result[:, 1] *= scale_y
    return result


def preview_to_image_points(
    points: Sequence[Sequence[float]],
    source_shape: tuple[int, int],
    preview_shape: tuple[int, int],
) -> np.ndarray:
    """Масштабує точки з координат preview назад у координати source."""
    src_h, src_w = source_shape
    preview_h, preview_w = preview_shape
    scale_x = src_w / max(preview_w, 1)
    scale_y = src_h / max(preview_h, 1)
    result = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    result[:, 0] *= scale_x
    result[:, 1] *= scale_y
    return result
