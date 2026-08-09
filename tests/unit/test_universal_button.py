"""Тести налаштувань і pipeline для Універсальної кнопки."""

import numpy as np

from config import app_settings
from processing import pipeline


def test_run_universal_returns_copy_when_no_step_is_enabled():
    image = np.zeros((8, 9, 3), dtype=np.uint8)

    result, status = pipeline.run_universal(image, {})

    assert status == "Не обрано жодного кроку"
    assert result is not image
    np.testing.assert_array_equal(result, image)


def test_run_universal_uses_fixed_order_and_classifier(monkeypatch):
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    events = []

    settings = {
        "universal_shadow_remove_enabled": True,
        "universal_color_cast_enabled": True,
        "universal_brightness_enabled": True,
        "universal_brightness_value": 0.2,
        "universal_contrast_enabled": True,
        "universal_contrast_value": 0.3,
        "universal_hdr_enabled": True,
        "universal_hdr_value": 0.4,
        "universal_sharpen_enabled": True,
        "universal_sharpen_value": 0.5,
        "universal_grayscale_enabled": True,
        "universal_white_background_enabled": True,
        "classify_bw_std_thresh": 18.0,
        "classify_edge_ratio_min": 0.08,
        "classify_line_count_min": 6,
    }

    def mark(name, delta=1):
        def operation(current, *args, **kwargs):
            events.append(name)
            return current + delta

        return operation

    def fake_classify(current, **kwargs):
        events.append(("classify", kwargs))
        return pipeline.DocType.COLOR_DOCUMENT.value

    def fake_shadow(current, is_color_document, settings):
        events.append("shadow_remove")
        assert is_color_document is True
        return current + 1, True

    def fake_color_cast(current):
        events.append("color_cast")
        return current + 1, True

    def fake_white_background(current, doc_type):
        events.append("white_background")
        assert doc_type == pipeline.DocType.COLOR_DOCUMENT.value
        return current + 1, True

    monkeypatch.setattr(pipeline, "run_classify", fake_classify)
    monkeypatch.setattr(pipeline, "run_shadow_remove_manual", fake_shadow)
    monkeypatch.setattr(pipeline.color_cast, "correct_color_cast", fake_color_cast)
    monkeypatch.setattr(pipeline, "run_brightness", mark("brightness"))
    monkeypatch.setattr(pipeline, "run_contrast_advanced", mark("contrast"))
    monkeypatch.setattr(pipeline.hdr, "apply", mark("hdr"))
    monkeypatch.setattr(pipeline.sharpen, "apply", mark("sharpen"))
    monkeypatch.setattr(pipeline, "run_grayscale", mark("grayscale"))
    monkeypatch.setattr(pipeline, "_apply_auto_white_background", fake_white_background)

    result, status = pipeline.run_universal(image, settings)

    assert events[0] == (
        "classify",
        {
            "bw_std_thresh": 18.0,
            "edge_ratio_min": 0.08,
            "line_count_min": 6,
        },
    )
    assert events[1:] == [
        "shadow_remove",
        "color_cast",
        "brightness",
        "contrast",
        "hdr",
        "sharpen",
        "grayscale",
        "white_background",
    ]
    assert result[0, 0, 0] == 8
    assert status.startswith("Універсальна: ")
    assert "тіні" in status
    assert "різкість=0.50" in status


def test_universal_settings_round_trip(tmp_path):
    settings = app_settings.load()
    settings.update(
        {
            "universal_shadow_remove_enabled": True,
            "universal_brightness_enabled": True,
            "universal_brightness_value": -0.25,
            "universal_contrast_enabled": True,
            "universal_contrast_value": 0.35,
            "universal_sharpen_enabled": True,
            "universal_sharpen_value": 0.6,
            "universal_hdr_enabled": True,
            "universal_hdr_value": 0.45,
            "universal_grayscale_enabled": True,
            "universal_white_background_enabled": True,
            "universal_color_cast_enabled": True,
        }
    )
    path = tmp_path / "settings.ini"

    app_settings.save(settings, path=str(path))
    loaded = app_settings.load(path=str(path))

    for key, value in settings.items():
        if key.startswith("universal_"):
            assert loaded[key] == value


def test_universal_handler_commits_one_result_or_reports_empty(monkeypatch):
    from gui.main_window import MainWindow

    window = MainWindow.__new__(MainWindow)
    window._base = np.zeros((8, 8, 3), dtype=np.uint8)
    window._orig = window._base.copy()
    window._settings = {"universal_sharpen_enabled": True}
    window._perspective_corners = object()
    window._btn_universal = object()

    pending = {}
    window._run_in_background = lambda work, done, button_to_lock=None: pending.update(
        work=work, done=done, button=button_to_lock
    )
    committed = []
    window._commit_base_result = lambda *args, **kwargs: committed.append((args, kwargs))
    statuses = []
    window._set_status = lambda status: statuses.append(status)

    monkeypatch.setattr(
        pipeline,
        "run_universal",
        lambda image, settings: (image + 1, "Універсальна: різкість=0.60"),
    )
    window._do_universal()
    assert pending["button"] is window._btn_universal
    pending["done"](pending["work"]())

    assert committed[0][0][1] == "Універсальна: різкість=0.60"
    assert committed[0][1] == {
        "autofix_applied": None,
        "update_before": False,
        "update_after": True,
    }

    committed.clear()
    window._settings = {}
    monkeypatch.setattr(
        pipeline,
        "run_universal",
        lambda image, settings: (image.copy(), "Не обрано жодного кроку"),
    )
    window._do_universal()
    pending["done"](pending["work"]())

    assert committed == []
    assert statuses[-1] == "Універсальна кнопка: не обрано жодного кроку в Налаштуваннях"
