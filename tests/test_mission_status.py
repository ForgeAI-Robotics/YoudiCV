from piper_handeye.mission_status import (
    append_status,
    normalize_flow_event,
    read_events,
)


def test_status_outbox_orders_events_and_reads_complete_records(tmp_path):
    path = tmp_path / "mission.jsonl"
    first = append_status(path, {
        "task_id": "task-a", "mission_id": "mission-a",
        "phase": "pressing", "outcome": "active",
    })
    offset = path.stat().st_size
    second = append_status(path, {
        "task_id": "task-a", "mission_id": "mission-a",
        "phase": "retry_required", "outcome": "active",
    })
    events, new_offset = read_events(path, offset)
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert new_offset == path.stat().st_size
    assert [event["phase"] for event in events] == ["retry_required"]


def test_status_reader_does_not_consume_an_incomplete_tail(tmp_path):
    path = tmp_path / "mission.jsonl"
    record = append_status(path, {"task_id": "task-b", "phase": "pressing"})
    offset = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b'{"partial":')
    events, next_offset = read_events(path, 0)
    assert events == [record]
    assert next_offset == offset


def test_flow_events_keep_retry_and_force_evidence_in_contract():
    status = normalize_flow_event({
        "event": "press_attempt_finished", "attempt": 2,
        "feedback_contact_confirmed": True,
        "bridge_status": "contact_confirmed",
        "encoder_error_rad": 0.004,
    })
    assert status["phase"] == "press_feedback_received"
    assert status["attempt"] == 2
    assert status["evidence"]["feedback_contact_confirmed"] is True
    assert status["evidence"]["encoder_error_rad"] == 0.004
