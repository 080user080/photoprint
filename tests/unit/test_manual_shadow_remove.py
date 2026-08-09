"""Тести окремої кнопки примусового видалення тіні."""

from types import SimpleNamespace

import numpy as np

from processing import pipeline


def test_run_shadow_remove_manual_always_uses_forced_algorithm(monkeypatch):
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    calls = {}

    def fake_remove_shadow(image, **kwargs):
        calls["image"] = image
        calls["kwargs"] = kwargs
        return image + 1

    def fail_auto_remove_shadow(*args, **kwargs):
        raise AssertionError("manual shadow removal must not run auto detection")

    monkeypatch.setattr(pipeline.shadow_remove, "remove_shadow", fake_remove_shadow)
    monkeypatch.setattr(pipeline.shadow_remove, "auto_remove_shadow", fail_auto_remove_shadow)

    result, applied = pipeline.run_shadow_remove_manual(
        image,
        is_color_document=True,
        settings={
            "shadow_coarse_blend_color": 0.35,
            "shadow_bgr_mode": True,
            "shadow_detect_threshold": 1.0,
            "shadow_uniformity_low": 0.99,
        },
    )

    assert applied is True
    assert calls["image"] is image
    assert calls["kwargs"] == {
        "is_color_document": True,
        "coarse_blend": 0.35,
        "bgr_mode": True,
    }
    np.testing.assert_array_equal(result, image + 1)


def test_shadow_remove_handler_classifies_in_background(monkeypatch):
    from gui.main_window import MainWindow

    window = MainWindow.__new__(MainWindow)
    window._base = np.zeros((10, 10, 3), dtype=np.uint8)
    window._orig = window._base.copy()
    window._settings = {
        "classify_bw_std_thresh": 17.0,
        "classify_edge_ratio_min": 0.07,
        "classify_line_count_min": 5,
    }
    window._perspective_corners = object()
    window._btn_shadow_remove = object()

    calls = {}

    def fake_classify(image, **kwargs):
        calls["classify"] = (image, kwargs)
        return pipeline.DocType.COLOR_DOCUMENT.value

    def fake_manual(image, is_color_document, settings):
        calls["manual"] = (image, is_color_document, settings)
        return image + 2, True

    monkeypatch.setattr(pipeline, "run_classify", fake_classify)
    monkeypatch.setattr(pipeline, "run_shadow_remove_manual", fake_manual)

    pending = {}
    window._run_in_background = lambda work, done, button_to_lock=None: pending.update(
        work=work, done=done, button=button_to_lock
    )
    committed = {}
    window._commit_base_result = lambda *args, **kwargs: committed.update(
        args=args, kwargs=kwargs
    )

    window._do_shadow_remove()

    assert pending["button"] is window._btn_shadow_remove
    result, had_shadow = pending["work"]()
    pending["done"]((result, had_shadow))

    assert calls["classify"][1] == {
        "bw_std_thresh": 17.0,
        "edge_ratio_min": 0.07,
        "line_count_min": 5,
    }
    assert calls["manual"][1] is True
    assert calls["manual"][2] == window._settings
    assert committed["args"][1] == "Видалення тіні застосовано"
    assert committed["kwargs"] == {
        "autofix_applied": None,
        "update_before": False,
        "update_after": True,
    }


def test_reset_progress_restores_the_locked_button_text():
    from gui.main_window import MainWindow

    class FakeButton:
        def __init__(self, text):
            self._text = text
            self._properties = {"_orig_text": text}

        def property(self, name):
            return self._properties.get(name)

        def setProperty(self, name, value):
            self._properties[name] = value

        def setEnabled(self, value):
            self.enabled = value

        def setText(self, value):
            self._text = value

        def text(self):
            return self._text

    window = MainWindow.__new__(MainWindow)
    window._progress = SimpleNamespace(setVisible=lambda value: None, setRange=lambda *args: None)
    window._set_buttons_enabled = lambda enabled: None
    button = FakeButton("🌗 Прибрати тінь")

    button.setText("⏳ Обробка…")
    window._reset_progress_ui(button)

    assert button.enabled is True
    assert button.text() == "🌗 Прибрати тінь"
    assert button.property("_orig_text") is None
