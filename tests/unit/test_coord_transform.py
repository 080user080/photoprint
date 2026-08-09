"""Тести перетворень координат source ↔ preview."""

import numpy as np

from utils.coord_transform import image_to_preview_points, preview_to_image_points


def test_image_to_preview_points_scales_axes_independently():
    points = image_to_preview_points(
        [[0, 0], [199, 99]], (100, 200), (50, 80)
    )
    np.testing.assert_allclose(points, [[0, 0], [79.6, 49.5]])


def test_preview_to_image_points_is_inverse_for_float_points():
    source = np.array([[0, 0], [199.25, 98.5]], dtype=np.float32)
    preview = image_to_preview_points(source, (100, 200), (50, 80))
    restored = preview_to_image_points(preview, (100, 200), (50, 80))
    np.testing.assert_allclose(restored, source)
