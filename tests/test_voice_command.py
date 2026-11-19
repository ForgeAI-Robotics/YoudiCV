import pytest

from piper_handeye.voice_command import (
    DecisionStatus,
    OneShotConsensusSession,
    NamedAssistantSession,
    VoiceCommand,
    VoiceCommandSession,
    VoiceIntent,
    parse_voice_command,
)


@pytest.fixture(autouse=True)
def isolate_mission_status_outbox(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(tmp_path / "mission-status.jsonl")
    )


def test_voice_dispatcher_propagates_production_config_to_closed_loop_batch(tmp_path):
    from run_voice_command_gateway import CommandDispatcher

    config = tmp_path / "robot.yaml"
    hold = tmp_path / "hold.yaml"
    dispatcher = CommandDispatcher(execute=False, config_path=config, hold_report=hold)
    argv = dispatcher._motion_argv(VoiceCommand(
        VoiceIntent.PRESS_BUTTON, "9", "9", button=9
    ))
    assert argv[1].endswith("run_button_batch.py")
    assert argv[argv.index("--config") + 1] == str(config.resolve())
    assert argv[argv.index("--hold-report") + 1] == str(hold.resolve())


def test_voice_dispatcher_accepts_direction_button_labels(tmp_path):
    from run_voice_command_gateway import CommandDispatcher

    dispatcher = CommandDispatcher(
        execute=False, config_path=tmp_path / "robot.yaml", hold_report=tmp_path / "hold.yaml"
    )
    for target in ("up", "down"):
        argv = dispatcher._motion_argv(VoiceCommand(
            VoiceIntent.PRESS_BUTTON, target, target, button=target
        ))
        assert argv[1].endswith("run_button_batch.py")
        assert argv[2] == target


def test_voice_trip_dry_run_selects_direction_from_test_floor(tmp_path):
    from run_voice_command_gateway import CommandDispatcher

    dispatcher = CommandDispatcher(
        execute=False, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    dispatcher.set_current_floor(16)
    result = dispatcher.start_trip(VoiceCommand(
        VoiceIntent.PRESS_BUTTON, "去一楼", "去一楼", button=1
    ))
    assert result == {
        "event": "dry_run", "trip_destination": 1,
        "current_floor": 16, "call_button": "down",
    }


def test_voice_trip_selects_up_when_target_is_above_current_floor(tmp_path, monkeypatch):
    from run_voice_command_gateway import CommandDispatcher

    class FakeProcess:
        pid = 123

        def poll(self):
            return None

    monkeypatch.setattr(
        "run_voice_command_gateway.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    dispatcher = CommandDispatcher(
        execute=True, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    dispatcher.set_current_floor(1)
    result = dispatcher.start_trip(VoiceCommand(
        VoiceIntent.PRESS_BUTTON, "送外卖到十五楼", "送外卖到十五楼", button=15
    ))
    assert result["call_button"] == "up"
    assert result["current_floor"] == 1
    assert result["trip_destination"] == 15
    assert result["argv"][2] == "up"


def test_voice_trip_rejects_same_current_and_target_floor(tmp_path):
    from run_voice_command_gateway import CommandDispatcher

    dispatcher = CommandDispatcher(
        execute=False, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    dispatcher.set_current_floor(8)
    with pytest.raises(RuntimeError, match="相同"):
        dispatcher.start_trip(VoiceCommand(
            VoiceIntent.PRESS_BUTTON, "去八楼", "去八楼", button=8
        ))


def test_voice_trip_requires_a_known_current_floor(tmp_path):
    from run_voice_command_gateway import CommandDispatcher

    dispatcher = CommandDispatcher(
        execute=False, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    with pytest.raises(RuntimeError, match="设定当前测试楼层"):
        dispatcher.start_trip(VoiceCommand(
            VoiceIntent.PRESS_BUTTON, "去一楼", "去一楼", button=1
        ))


def test_voice_trip_waits_for_square_between_direction_and_floor(tmp_path, monkeypatch):
    from run_voice_command_gateway import CommandDispatcher

    class FakeProcess:
        pid = 123

        def __init__(self):
            self.return_code = None

        def poll(self):
            return self.return_code

    processes = [FakeProcess(), FakeProcess()]
    monkeypatch.setattr(
        "run_voice_command_gateway.subprocess.Popen",
        lambda *args, **kwargs: processes.pop(0),
    )
    dispatcher = CommandDispatcher(
        execute=True, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    dispatcher.set_current_floor(16)
    started = dispatcher.start_trip(VoiceCommand(
        VoiceIntent.PRESS_BUTTON, "去一楼", "去一楼", button=1
    ))
    mission_id = started["mission_id"]
    assert started["event"] == "motion_started"
    assert started["argv"][2] == "down"
    processes_ref = [dispatcher.active]
    processes_ref[0].return_code = 0
    waiting = dispatcher.refresh()
    assert waiting["event"] == "trip_waiting_for_square"
    assert dispatcher.active is None

    resumed = dispatcher.resume_trip()
    assert resumed["event"] == "motion_started"
    assert resumed["mission_id"] == mission_id
    assert resumed["argv"][2] == "1"
    dispatcher.active.return_code = 0
    completed = dispatcher.refresh()
    assert completed["event"] == "motion_completed"
    assert completed["trip_stage"] == "floor"
    assert dispatcher.current_floor == 1
    assert dispatcher.trip_stage is None


def test_dispatchers_cannot_run_robot_tasks_concurrently(tmp_path, monkeypatch):
    from run_voice_command_gateway import CommandDispatcher

    class FakeProcess:
        pid = 321

        def __init__(self):
            self.return_code = None

        def poll(self):
            return self.return_code

    processes = []

    def launch(*_args, **_kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setenv(
        "YOUXIAOMEI_ROBOT_MOTION_LOCK", str(tmp_path / "robot-motion.lock")
    )
    monkeypatch.setattr("run_voice_command_gateway.subprocess.Popen", launch)
    first = CommandDispatcher(
        execute=True, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    second = CommandDispatcher(
        execute=True, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    command = VoiceCommand(VoiceIntent.PRESS_BUTTON, "8", "8", button=8)
    first.dispatch_motion(command)
    with pytest.raises(RuntimeError, match="占用机械臂任务执行权"):
        second.dispatch_motion(command)
    processes[0].return_code = 0
    first.refresh()
    second.dispatch_motion(command)
    processes[1].return_code = 0
    second.refresh()


def test_dispatcher_streams_child_arm_status_with_task_identity(tmp_path, monkeypatch):
    from piper_handeye.mission_status import append_status
    from run_voice_command_gateway import CommandDispatcher

    class FakeProcess:
        pid = 654

        def __init__(self):
            self.return_code = None

        def poll(self):
            return self.return_code

    process = FakeProcess()
    outbox = tmp_path / "mission.jsonl"
    launch_env = {}
    monkeypatch.setenv("YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(outbox))
    def launch(*_args, **kwargs):
        launch_env.update(kwargs["env"])
        return process

    monkeypatch.setattr("run_voice_command_gateway.subprocess.Popen", launch)
    dispatcher = CommandDispatcher(
        execute=True, config_path=tmp_path / "robot.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    started = dispatcher.dispatch_motion(VoiceCommand(
        VoiceIntent.PRESS_BUTTON, "按 8 楼", "按 8 楼", button=8
    ))
    assert started["task_id"]
    assert started["mission_id"] == started["task_id"]
    assert launch_env["YOUXIAOMEI_TASK_ID"] == started["task_id"]
    assert launch_env["YOUXIAOMEI_MISSION_ID"] == started["mission_id"]
    append_status(outbox, {
        "task_id": started["task_id"],
        "mission_id": started["mission_id"],
        "phase": "retry_required", "outcome": "active",
    })
    progress = dispatcher.refresh()
    assert progress["event"] == "mission_progress"
    assert progress["mission_status_events"][-1]["phase"] == "retry_required"
    process.return_code = 0
    completed = dispatcher.refresh()
    assert completed["event"] == "motion_completed"
    assert completed["task_id"] == started["task_id"]


def test_bare_arabic_number_selects_button_but_requires_confirmation():
    parsed = parse_voice_command("15")
    assert parsed.intent is VoiceIntent.PRESS_BUTTON
    assert parsed.button == 15
    assert parsed.bare_number

    session = VoiceCommandSession()
    decision = session.submit("15", confidence=0.96, now=10.0)
    assert decision.status is DecisionStatus.CONFIRMATION_REQUIRED
    assert decision.command.button == 15


def test_supported_button_phrasings_map_to_same_command():
    phrases = ("十五", "十五楼", "第十五层", "按十五", "button 15", "15 floor")
    for phrase in phrases:
        parsed = parse_voice_command(phrase)
        assert parsed.intent is VoiceIntent.PRESS_BUTTON, phrase
        assert parsed.button == 15, phrase


def test_financial_and_traditional_chinese_are_supported():
    assert parse_voice_command("按壹拾陸號").button == 16


def test_multiple_different_numbers_are_rejected():
    parsed = parse_voice_command("按 1、5、16")
    assert parsed.intent is VoiceIntent.UNKNOWN
    assert parsed.reason == "ambiguous_button_numbers"


def test_out_of_range_button_is_rejected():
    parsed = parse_voice_command("二十楼")
    assert parsed.intent is VoiceIntent.UNKNOWN


def test_confirmation_executes_pending_command_only_once():
    session = VoiceCommandSession()
    assert session.submit("按 12", now=10.0).status is DecisionStatus.CONFIRMATION_REQUIRED
    executed = session.submit("确认", now=11.0)
    assert executed.status is DecisionStatus.EXECUTE
    assert executed.command.button == 12
    assert session.submit("确认", now=11.1).status is DecisionStatus.REJECTED
    assert session.submit("12", now=11.2).status is DecisionStatus.REJECTED


def test_confirmation_expires_closed():
    session = VoiceCommandSession(confirmation_timeout_sec=2.0)
    session.submit("8", now=10.0)
    result = session.submit("确认", now=12.1)
    assert result.status is DecisionStatus.REJECTED
    assert session.pending is None


def test_low_confidence_bare_number_is_rejected():
    session = VoiceCommandSession()
    result = session.submit("15", confidence=0.84)
    assert result.status is DecisionStatus.REJECTED


def test_stop_is_immediate_even_at_zero_confidence_and_clears_pending():
    session = VoiceCommandSession()
    session.submit("15", confidence=0.99)
    result = session.submit("立即停止", confidence=0.0)
    assert result.status is DecisionStatus.STOP
    assert session.pending is None


def test_different_pending_command_cannot_be_silently_replaced():
    session = VoiceCommandSession()
    session.submit("15", now=10.0)
    result = session.submit("16", now=10.5)
    assert result.status is DecisionStatus.REJECTED
    assert session.pending.button == 15


def test_return_requires_confirmation_and_cancel_clears_it():
    session = VoiceCommandSession()
    result = session.submit("返回原点", now=10.0)
    assert result.status is DecisionStatus.CONFIRMATION_REQUIRED
    assert result.command.intent is VoiceIntent.RETURN_ORIGIN
    assert session.submit("取消", now=11.0).status is DecisionStatus.CANCELLED
    assert session.pending is None


def test_negative_phrase_cancels_instead_of_proposing_motion():
    session = VoiceCommandSession()
    session.submit("返回原点", now=10.0)
    result = session.submit("不要返回原点", now=11.0)
    assert result.status is DecisionStatus.CANCELLED
    assert session.pending is None


def test_one_shot_consensus_executes_matching_internal_decodes():
    session = OneShotConsensusSession()
    result = session.submit("十五号按钮", 0.42, "十五", 0.40, now=10.0)
    assert result.status is DecisionStatus.EXECUTE
    assert result.command.button == 15


def test_one_shot_consensus_rejects_different_buttons():
    session = OneShotConsensusSession()
    result = session.submit("十五", 0.7, "十四", 0.8, now=10.0)
    assert result.status is DecisionStatus.REJECTED


def test_one_shot_consensus_rejects_weak_agreement():
    session = OneShotConsensusSession()
    result = session.submit("十五", 0.24, "十五楼", 0.9, now=10.0)
    assert result.status is DecisionStatus.REJECTED


def test_one_shot_stop_is_fail_safe_if_either_decode_hears_stop():
    session = OneShotConsensusSession()
    result = session.submit("无关内容", 0.1, "立即停止", 0.2, now=10.0)
    assert result.status is DecisionStatus.STOP


def test_one_shot_consensus_suppresses_immediate_duplicate():
    session = OneShotConsensusSession()
    assert session.submit("15", 0.6, "十五", 0.6, now=10.0).status is DecisionStatus.EXECUTE
    assert session.submit("15", 0.6, "十五", 0.6, now=11.0).status is DecisionStatus.REJECTED


def test_addressed_commands_support_numbers_and_traditional_text():
    for phrase in (
        "送外卖到15楼", "送外卖到十五层", "送外賣到十五樓",
    ):
        parsed = parse_voice_command(phrase, require_wake_word=True)
        assert parsed.intent is VoiceIntent.PRESS_BUTTON
        assert parsed.button == 15
    assert parse_voice_command("送外卖，到第15楼", require_wake_word=True).button == 15
    assert parse_voice_command(
        "去一楼", require_wake_word=True, wake_confirmed=True
    ).button == 1


def test_address_gate_rejects_background_and_ambiguous_sentences():
    for phrase in (
        "十五楼", "15", "我昨天在十五楼吃饭", "他说机器人十五楼",
        "机器人我昨天在十五楼吃饭", "机器人十五楼还是十六楼",
        "机器人", "机器人二十楼", "机器人返回原点以后去吃饭",
        "送外卖到十五楼吃饭", "他说送外卖到十五楼", "送外卖到十五楼还是十六楼",
        "送外卖", "送外卖十五楼", "送外卖到20楼", "送外卖到二十楼",
    ):
        assert parse_voice_command(phrase, require_wake_word=True).intent is VoiceIntent.UNKNOWN


def test_both_decodes_need_address_and_no_wake_window_is_opened():
    session = OneShotConsensusSession(require_wake_word=True)
    assert session.submit("送外卖到十五楼", .9, "十五楼", .9).status is DecisionStatus.REJECTED
    result = session.submit("送外卖到十五楼", .9, "送外卖到15楼", .9, now=10)
    assert result.status is DecisionStatus.EXECUTE
    assert result.command.button == 15
    assert session.submit("16", .9, "十六楼", .9, now=20).status is DecisionStatus.REJECTED


def test_address_gate_preserves_stop_cancel_and_return():
    session = OneShotConsensusSession(require_wake_word=True)
    for phrase in ("停止", "送外卖停止", "急停"):
        assert session.submit(phrase, .1, "无关内容", .1).status is DecisionStatus.STOP
    assert session.submit("取消", .9, "无关内容", .9).status is DecisionStatus.CANCELLED
    assert session.submit("返回原点", .9, "返回原点", .9).status is DecisionStatus.REJECTED
    assert session.submit("送外卖返回原点", .9, "送外卖返回原点", .9).status is DecisionStatus.EXECUTE


def test_legacy_confirmation_only_accepts_addressed_proposals():
    session = VoiceCommandSession(require_wake_word=True)
    assert session.submit("15", now=1).status is DecisionStatus.REJECTED
    assert session.submit("确认", now=2).status is DecisionStatus.REJECTED
    assert session.submit("送外卖到15楼", now=3).status is DecisionStatus.CONFIRMATION_REQUIRED
    assert session.submit("确认", now=4).status is DecisionStatus.EXECUTE


def test_named_assistant_acknowledges_then_executes_next_command():
    session = NamedAssistantSession("优小妹", wake_window_sec=8.0)
    wake = session.submit("优小妹", .9, "优小妹。", .9, now=10.0)
    assert wake.awakened
    assert wake.wake_message == "在的"
    assert wake.decision is None
    command = session.submit(
        "送外卖到十五楼", .8, "送外卖到15楼", .8, now=12.0
    )
    assert not command.awakened
    assert command.decision.status is DecisionStatus.EXECUTE
    assert command.decision.command.button == 15


def test_named_assistant_explains_command_after_wake_timeout():
    session = NamedAssistantSession("优小妹", wake_window_sec=3.0)
    session.submit("優小妹", .9, "优小妹", .9, now=10.0)
    result = session.submit("送外卖到十五楼", .9, "送外卖到十五楼", .9, now=13.1)
    assert not result.awakened
    assert result.decision.status is DecisionStatus.REJECTED
    assert "优小妹" in result.decision.message


def test_named_assistant_explains_valid_command_without_active_wake():
    session = NamedAssistantSession("优小妹", wake_window_sec=3.0)
    result = session.submit(
        "送外卖到八楼", .9, "送外卖到8楼", .9, now=10.0
    )
    assert result.decision.status is DecisionStatus.REJECTED
    assert "优小妹" in result.decision.message
    assert result.decision.command.button == 8


def test_named_same_utterance_command_requires_both_decodes_to_hear_name():
    session = NamedAssistantSession()
    result = session.submit(
        "优小妹送外卖到十五楼", .9, "送外卖到十五楼", .9, now=10.0
    )
    assert result.wake_message == "在的"
    assert result.decision.status is DecisionStatus.REJECTED


def test_named_assistant_stop_remains_available_while_asleep():
    result = NamedAssistantSession().submit(
        "无关内容", .1, "立即停止", .1, now=10.0
    )
    assert result.decision.status is DecisionStatus.STOP
