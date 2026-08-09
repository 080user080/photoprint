"""Тести чистих рішень pipeline без запуску важких обробників."""

from processing.doc_classifier import CAPTURE_PHONE, CAPTURE_SCREEN
from processing.pipeline import DocType, decide_shadow_remove


def test_screen_capture_is_always_skipped():
    assert decide_shadow_remove(
        DocType.BW_DOCUMENT.value, CAPTURE_SCREEN, 0.9, False, False
    ) == (False, "screen_capture")


def test_bw_document_with_mid_uniformity_runs_without_face():
    assert decide_shadow_remove(
        DocType.BW_DOCUMENT.value, "unknown", 0.4, False, False
    ) == (True, "doc_type=bw_document")


def test_bw_document_color_cast_overrides_mid_uniformity():
    assert decide_shadow_remove(
        DocType.BW_DOCUMENT.value, "unknown", 0.4, True, True
    ) == (True, "bw_document + color_cast")


def test_phone_photo_uses_reduced_uniformity_threshold():
    should_run, reason = decide_shadow_remove(
        DocType.PHOTO.value,
        CAPTURE_PHONE,
        0.56,
        False,
        False,
        {"shadow_uniformity_photo_high": 0.65},
    )
    assert should_run is True
    assert reason == "photo uniformity=0.56>0.55"
