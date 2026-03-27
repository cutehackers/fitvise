from app.domain.services.session_service import SessionConfig, SessionService


def test_turn_window_trims_old_non_system_messages():
    service = SessionService(
        SessionConfig(turns_window=1, max_messages_per_session=10)
    )
    session_id = "s1"

    service.add_system_message(session_id, "sys")
    service.add_user_message(session_id, "u1")
    service.add_assistant_message(session_id, "a1")
    service.add_user_message(session_id, "u2")
    service.add_assistant_message(session_id, "a2")

    history = service.get_session_history(session_id)
    messages = history.messages

    assert len(messages) == 3  # sys + latest user/assistant
    assert messages[0].content == "sys"
    assert messages[1].content == "u2"
    assert messages[2].content == "a2"


def test_max_messages_trims_oldest_non_system_first():
    service = SessionService(
        SessionConfig(turns_window=5, max_messages_per_session=3)
    )
    session_id = "s2"

    service.add_system_message(session_id, "sys")
    service.add_user_message(session_id, "u1")
    service.add_user_message(session_id, "u2")
    service.add_user_message(session_id, "u3")

    history = service.get_session_history(session_id)
    messages = history.messages

    assert len(messages) == 3
    assert messages[0].content == "sys"
    assert messages[1].content == "u2"
    assert messages[2].content == "u3"
