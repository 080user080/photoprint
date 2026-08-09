"""Тести чистого осьовирівняного кадрування Кроку 6."""

import numpy as np

from processing import pipeline


def test_run_crop_rect_preserves_selected_pixels_without_padding():
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    corners = np.array([
        [1, 1], [4, 1], [4, 3], [1, 3],
    ], dtype=np.float32)

    result = pipeline.run_crop_rect(image, corners)

    assert result.shape == (3, 4, 3)
    np.testing.assert_array_equal(result, image[1:4, 1:5])


def test_run_crop_rect_fills_area_outside_image_white():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :] = [10, 20, 30]
    corners = np.array([
        [-1, -1], [2, -1], [2, 1], [-1, 1],
    ], dtype=np.float32)

    result = pipeline.run_crop_rect(image, corners)

    assert result.shape == (3, 4, 3)
    np.testing.assert_array_equal(result[:2, :1], 255)
    np.testing.assert_array_equal(result[1:3, 1:], image[:2, :3])


def test_run_crop_rect_rejects_wrong_corner_count():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    try:
        pipeline.run_crop_rect(image, np.zeros((3, 2), dtype=np.float32))
    except ValueError as exc:
        assert "4 точки" in str(exc)
    else:
        raise AssertionError("Очікувався ValueError для не чотирьох точок")


def test_crop_pin_perspective_keeps_crop_canvas_size():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :30] = [10, 20, 30]
    image[:, 30:] = [200, 210, 220]
    corners = np.array([
        [8, 6], [59, 0], [59, 39], [0, 39],
    ], dtype=np.float32)

    result = pipeline.run_crop_pin_perspective(image, corners)

    assert result.shape == image.shape


def test_crop_pin_identity_preserves_pixels_and_size():
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    corners = np.array([
        [0, 0], [5, 0], [5, 4], [0, 4],
    ], dtype=np.float32)

    result = pipeline.run_crop_pin_perspective(image, corners)

    assert result.shape == image.shape
    np.testing.assert_array_equal(result, image)


def test_detect_document_bounds_returns_axis_aligned_box():
    image = np.full((100, 140, 3), 255, dtype=np.uint8)
    image[20:80, 30:110] = 30

    bounds = pipeline.detect_document_bounds(image)

    assert bounds is not None
    assert bounds.shape == (4, 2)
    assert bounds[0, 0] >= 25
    assert bounds[0, 1] >= 15
    assert bounds[2, 0] <= 115
    assert bounds[2, 1] <= 85
