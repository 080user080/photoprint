from gui.edit_state import EditMode, EditSession


def test_edit_session_is_idle_by_default():
    session = EditSession()
    assert session.mode is EditMode.IDLE
    assert session.active is False


def test_edit_session_reports_active_modes():
    session = EditSession(EditMode.CROP)
    assert session.active is True
    session.mode = EditMode.DESKEW_PENDING
    assert session.active is True
