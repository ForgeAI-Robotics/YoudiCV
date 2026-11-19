from piper_handeye.assistant_gui_events import AssistantEventPresenter, FloorEstimate


def _text(messages):
    return [(item.speaker, item.text) for item in messages]


def test_chat_presenter_maps_required_live_messages():
    presenter = AssistantEventPresenter("优小妹")
    assert _text(presenter.consume({"event": "transcript", "text": "优小妹"})) == [
        ("用户", "优小妹")
    ]
    assert _text(presenter.consume({"event": "assistant_wake"})) == [
        ("优小妹", "在的")
    ]
    assert _text(presenter.consume({"event": "assistant_wake_cancelled"})) == [
        ("优小妹", "好的")
    ]
    assert _text(presenter.consume({"event": "assistant_wake_prompt"})) == [
        ("优小妹", "请给我任务")
    ]
    assert _text(presenter.consume({"event": "assistant_wake_timeout"})) == [
        ("优小妹", "任务未下达")
    ]
    assert _text(presenter.consume({"event": "motion_started"})) == [
        ("优小妹", "正在执行")
    ]
    assert _text(presenter.consume({"phase": "visual_safety_paused"})) == [
        ("优小妹", "安全监测触发暂停，原因待核实")
    ]
    assert _text(presenter.consume({
        "phase": "visual_safety_paused",
        "value": {"reason": "rgb_scene_change_unverified_near_panel"},
    })) == [("优小妹", "视觉画面变化，尚未确认异物，安全暂停")]
    assert _text(presenter.consume({"phase": "flow:retry_required"})) == [
        ("优小妹", "尝试再次按压")
    ]
    assert _text(presenter.consume({"event": "motion_completed", "return_code": 0})) == [
        ("优小妹", "机械臂按钮操作已完成")
    ]


def test_nonzero_task_exit_is_reported_as_failure():
    presenter = AssistantEventPresenter()
    presenter.consume({"event": "motion_started"})
    assert _text(presenter.consume({"event": "motion_completed", "return_code": 2})) == [
        ("优小妹", "任务失败")
    ]
    assert not presenter.task_active


def test_abort_and_return_clears_chat_task_state():
    presenter = AssistantEventPresenter()
    presenter.consume({"event": "motion_started"})
    assert _text(presenter.consume({"event": "abort_return_started"})) == [
        ("优小妹", "正在中止任务并返回原位")
    ]
    assert not presenter.task_active


def test_rejection_reason_is_visible_in_chat():
    presenter = AssistantEventPresenter()
    assert _text(presenter.consume({
        "event": "decision",
        "status": "rejected",
        "message": "两路内部识别结果不一致",
    })) == [("优小妹", "两路内部识别结果不一致")]


def test_manual_and_voice_control_events_are_presented():
    presenter = AssistantEventPresenter()
    assert _text(presenter.consume({
        "event": "manual_command", "floor": 9, "text": "送外卖到9楼"
    })) == [("用户", "送外卖到9楼")]
    assert _text(presenter.consume({
        "event": "voice_state", "enabled": True
    })) == [("优小妹", "语音呼叫已开启")]
    assert _text(presenter.consume({
        "event": "voice_state", "enabled": False
    })) == [("优小妹", "语音呼叫已关闭")]
    assert _text(presenter.consume({
        "event": "manual_command_rejected", "message": "当前任务正在执行"
    })) == [("优小妹", "当前任务正在执行")]


def test_floor_estimate_updates_only_after_target_button_off_confirmation():
    estimate = FloorEstimate(execute=True)
    assert "未知" in estimate.consume({})
    estimate.consume({
        "event": "motion_started",
        "argv": ["python", "/project/run_button_batch.py", "15"],
    })
    assert "正在前往15 楼" in estimate.consume({})
    arrived = estimate.consume({
        "phase": "flow:button_became_off",
        "value": {"event": "button_became_off", "position": "waiting"},
    })
    assert "15 楼" in arrived
    completed = estimate.consume({
        "event": "motion_completed", "button": 15, "return_code": 0,
    })
    assert "15 楼" in completed


def test_up_down_call_does_not_invent_a_numeric_floor():
    estimate = FloorEstimate(execute=True)
    estimate.consume({"event": "motion_started", "argv": [
        "python", "/project/run_button_batch.py", "up",
    ]})
    assert "上行呼梯" in estimate.consume({})
    arrived = estimate.consume({"phase": "flow:button_became_off"})
    assert "未知" in arrived
    assert "没有具体楼层读数" in arrived


def test_preview_mode_never_claims_a_current_floor():
    estimate = FloorEstimate(execute=False)
    assert estimate.consume({"event": "motion_completed", "return_code": 0}) == (
        "当前楼层估计：未知（预览模式不驱动机械臂）"
    )


def test_manual_floor_is_starting_state_then_automatic_tracking_resumes():
    estimate = FloorEstimate(execute=True)
    assert estimate.set_manual_floor("8") == "当前楼层估计：8 楼（手动测试值）"
    moving = estimate.consume({
        "event": "motion_started", "argv": [
            "python", "/project/run_button_batch.py", "15",
        ],
    })
    assert "未知（正在前往15 楼" in moving
    assert "上次确认 8 楼" in moving


def test_manual_floor_override_validates_range():
    estimate = FloorEstimate(execute=True)
    for invalid in ("0", "17", "abc", "2.5"):
        try:
            estimate.set_manual_floor(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid floor to be rejected: {invalid}")


def test_staged_trip_floor_display_and_waiting_checkpoint():
    estimate = FloorEstimate(execute=True)
    estimate.set_manual_floor(16)
    estimate.consume({
        "event": "trip_waiting_for_square", "from_floor": 16, "button": 1,
    })
    waiting = estimate.set_manual_floor(None)
    assert "16 楼" in waiting
    assert "等待驶入电梯" in waiting
    estimate.set_manual_floor(None)
    estimate.consume({
        "event": "motion_started",
        "argv": ["python", "/project/run_button_batch.py", "1"],
    })
    arrived = estimate.consume({
        "event": "motion_completed", "button": 1,
        "trip_stage": "floor", "return_code": 0,
    })
    assert "1 楼" in arrived
    assert "确认到站" in arrived
