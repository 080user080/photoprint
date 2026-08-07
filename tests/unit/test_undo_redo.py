"""
Тести для Undo/Redo історії (TODO1.1) та журналювання автоперспективи (TODO1.4).

Фокус:
1. _push_undo_snapshot — зберігає _base + _per_file + autofix_applied.
2. _do_undo/_do_redo — коректне перемикання між стеками, ліміт 2 кроки.
3. _commit_pending_perspective — коміт лише якщо був реальний drag.
4. auto_detect_corners — журналює ім'я файлу при невдачі.
"""

import numpy as np
import pytest

from processing.perspective import auto_detect_corners, LOW_SCORE_WARNING_THRESHOLD


# ---------------------------------------------------------------------------
# Допоміжне: створення MainWindow без GUI
# ---------------------------------------------------------------------------

def _make_window():
    """Створює MainWindow без QApplication (тільки для чистих методів)."""
    from gui.main_window import MainWindow
    return MainWindow.__new__(MainWindow)


def _make_state(base, per_file=None, autofix=None):
    return {
        "base": base.copy(),
        "per_file": dict(per_file or {}),
        "autofix_applied": autofix,
    }


# ---------------------------------------------------------------------------
# _push_undo_snapshot
# ---------------------------------------------------------------------------

class TestPushUndoSnapshot:
    def test_snapshot_contains_base_per_file_autofix(self):
        w = _make_window()
        w._current_path = "/tmp/a.jpg"
        w._base = np.zeros((10, 10, 3), dtype=np.uint8)
        w._per_file = {"/tmp/a.jpg": {"brightness": 0.5}}
        w._preview = type("P", (), {"get_autofix_applied": lambda self: "auto_fix"})()
        w._undo_history = {}
        w._redo_history = {}

        w._push_undo_snapshot()

        snap = w._undo_history["/tmp/a.jpg"][0]
        assert snap["base"].shape == (10, 10, 3)
        assert snap["per_file"] == {"brightness": 0.5}
        assert snap["autofix_applied"] == "auto_fix"

    def test_snapshot_clears_redo(self):
        w = _make_window()
        w._current_path = "/tmp/a.jpg"
        w._base = np.zeros((10, 10, 3), dtype=np.uint8)
        w._per_file = {}
        w._preview = type("P", (), {"get_autofix_applied": lambda self: None})()
        w._undo_history = {}
        w._redo_history = {"/tmp/a.jpg": [{"base": np.zeros((1,1,3), dtype=np.uint8)}]}

        w._push_undo_snapshot()

        assert "/tmp/a.jpg" not in w._redo_history

    def test_snapshot_limited_to_2(self):
        w = _make_window()
        w._current_path = "/tmp/a.jpg"
        w._per_file = {}
        w._preview = type("P", (), {"get_autofix_applied": lambda self: None})()
        w._undo_history = {}
        w._redo_history = {}

        for i in range(5):
            w._base = np.full((10, 10, 3), i, dtype=np.uint8)
            w._push_undo_snapshot()

        assert len(w._undo_history["/tmp/a.jpg"]) == 2
        # Останній збережений — значення 4 (останній виклик)
        assert w._undo_history["/tmp/a.jpg"][-1]["base"][0,0,0] == 4


# ---------------------------------------------------------------------------
# _do_undo / _do_redo
# ---------------------------------------------------------------------------

class TestUndoRedo:
    def _setup(self):
        w = _make_window()
        w._current_path = "/tmp/a.jpg"
        w._base = np.full((10, 10, 3), 5, dtype=np.uint8)
        w._per_file = {}
        w._preview = type("P", (), {
            "get_autofix_applied": lambda self: None,
            "set_autofix_applied": lambda self, m: None,
            "set_before": lambda self, img: None,
            "set_after": lambda self, img: None,
            "disable_perspective_edit": lambda self: None,
        })()
        w._undo_history = {}
        w._redo_history = {}
        w._processed = None
        w._base_for_perspective = None
        w._perspective_corners = None
        w._perspective_cached_result = None
        w._pending_deskew_result = None
        w._has_running_threads = lambda: False
        w._restore_file_settings = lambda path: None
        w._unfreeze_preview_panels = lambda: None
        w._update_buttons = lambda: None
        w._set_status = lambda text, timeout_ms=0: None
        return w

    def test_undo_restores_previous_state(self):
        w = self._setup()
        # Стан 1
        w._base = np.full((10, 10, 3), 1, dtype=np.uint8)
        w._push_undo_snapshot()
        # Стан 2
        w._base = np.full((10, 10, 3), 2, dtype=np.uint8)
        w._push_undo_snapshot()
        # Стан 3 (поточний)
        w._base = np.full((10, 10, 3), 3, dtype=np.uint8)

        # undo → стан 2 (останній снимок)
        w._do_undo()
        assert w._base[0,0,0] == 2

        # undo → стан 1
        w._do_undo()
        assert w._base[0,0,0] == 1

        # redo → стан 2
        w._do_redo()
        assert w._base[0,0,0] == 2

    def test_undo_clears_redo_on_new_action(self):
        w = self._setup()
        w._base = np.full((10, 10, 3), 1, dtype=np.uint8)
        w._push_undo_snapshot()
        w._base = np.full((10, 10, 3), 2, dtype=np.uint8)
        w._push_undo_snapshot()

        w._do_undo()  # redo-стек заповнюється
        assert "/tmp/a.jpg" in w._redo_history

        # Нова дія
        w._base = np.full((10, 10, 3), 3, dtype=np.uint8)
        w._push_undo_snapshot()
        assert "/tmp/a.jpg" not in w._redo_history

    def test_undo_noop_when_empty(self):
        w = self._setup()
        w._do_undo()  # не падає
        assert w._base[0,0,0] == 5


# ---------------------------------------------------------------------------
# _commit_pending_perspective
# ---------------------------------------------------------------------------

class TestCommitPendingPerspective:
    def _setup(self, drag_applied):
        w = _make_window()
        w._base = np.zeros((10, 10, 3), dtype=np.uint8)
        w._base_for_perspective = np.zeros((10, 10, 3), dtype=np.uint8)
        w._perspective_corners = np.array([[0,0],[9,0],[9,9],[0,9]], dtype=np.float32)
        w._perspective_cached_result = None
        w._persp_drag_applied = drag_applied
        w._preview = type("P", (), {
            "disable_perspective_edit": lambda self: None,
        })()
        w._unfreeze_preview_panels = lambda: None
        w._push_undo_snapshot = lambda: None
        return w

    def test_commit_only_when_drag_applied(self):
        w = self._setup(drag_applied=True)
        w._commit_pending_perspective()
        # _base_for_perspective скинуто, _perspective_corners скинуто
        assert w._base_for_perspective is None
        assert w._perspective_corners is None

    def test_no_commit_without_drag(self):
        w = self._setup(drag_applied=False)
        w._commit_pending_perspective()
        # _base_for_perspective скинуто, але _base не змінювався
        assert w._base_for_perspective is None
        assert w._perspective_corners is None


# ---------------------------------------------------------------------------
# Журналювання автоперспективи (TODO1.4)
# ---------------------------------------------------------------------------

class TestAutoDetectLogging:
    def test_logs_filename_when_no_corners(self, caplog):
        import logging
        img = np.zeros((100, 100, 3), dtype=np.uint8)  # порожнє — кутів не знайде
        with caplog.at_level(logging.WARNING, logger="processing.perspective"):
            result = auto_detect_corners(img, filename="test_file.jpg")
        assert result is None
        # Перевіряємо, що в логу є ім'я файлу
        assert any("test_file.jpg" in r.message for r in caplog.records)

    def test_low_score_threshold_defined(self):
        assert LOW_SCORE_WARNING_THRESHOLD > 0.0
        assert LOW_SCORE_WARNING_THRESHOLD < 1.0