# Local voice-command conversion

Microphone ASR applies a second-order Butterworth low-pass at 4000 Hz to
each 16 kHz utterance before either decoding pass. Original diagnostic audio
is retained unchanged. Set `--lowpass-hz 0` to bypass, or supply another cutoff
below 8000 Hz. This reduces high-frequency noise; it does not prevent silence
hallucinations or establish that speech is an intentional robot command.

`run_voice_command_gateway.py` converts Whisper/SenseVoice transcripts into a
small allow-list of Piper operations. It accepts plain text or JSON Lines from
any ASR process. `run_voice_microphone.py` adds offline Mandarin recognition
from the DOV USB microphone using the local CUDA build of whisper.cpp.

## 优小妹实时对话 GUI

`run_voice_assistant_gui.py` 默认使用本地 sherpa-onnx SenseVoiceSmall int8，并把识别文本和生产任务
事件显示成“用户：/优小妹：”对话。说出中文“一二三”后，界面回复“在的”；静默 10 秒提示“请给我任务”，静默 20 秒清除唤醒并提示“任务未下达”。说“没事了”可立即取消本次唤醒。任务开始执行时语音识别自动关闭。Silero VAD 使用 1.0 秒句尾静音，允许更长的词间停顿。
命令窗口；随后说“送外卖到十五楼”会解析为按钮 15。真实执行仍复用
`run_voice_streaming.py -> CommandDispatcher -> run_button_batch.py ->
run_button_light_closed_loop.py`，不会绕过按钮灯态和运动安全闭环。

桌面有两个入口：

- `优小妹语音助手（安全预览）`：识别和显示完整，但不发送机器人命令。
- `优小妹语音助手（机器人执行）`：带现有的两项物理执行确认参数。

GUI 将以下真实事件映射为固定反馈：

- 任务子进程启动：`正在执行`
- 视觉安全停止桥确认暂停：`检测到异物，暂停任务`
- 障碍清除并恢复：`异物已移除，继续任务`
- 按钮灯态为 OFF、状态机进入下一次尝试：`尝试再次按压`
- 任务退出码为 0：`任务已完成`
- 非零退出码或安全监督失败：`任务失败`

关闭 GUI 会向语音后端发送 SIGINT；若任务仍在执行，后端先调用独立停止桥并
终止任务进程组，再关闭麦克风和流式识别器。GUI 本身不直接持有机器人接口。

GUI 启动后语音呼叫默认关闭。聊天记录上方可以输入 `1`–`16` 的目标楼层并点击
“执行任务”（也可按回车）；手动任务与语音任务共用同一个 `CommandDispatcher`，
所以任务进行中不能重复启动另一个任务。点击“开启语音”后才处理唤醒词和后续
本地流式识别；再次点击“关闭语音”会停止识别并清空关闭期间形成的音频帧，
避免重新开启时执行旧语音。

语音输入由 sherpa-onnx Silero VAD 切句，完整短句进入常驻 SenseVoiceSmall，
GUI 只显示唤醒词、任务和控制意图。被过滤结果仍写入 `voice_gui_runs/*.jsonl`。
当前模式按要求没有 Whisper 后备，也没有双模型复核；因此只允许准确唤醒后的完整
“送外卖到 1–16 楼”原始识别句式，不使用同音词表或缺词补全。模型没有校准置信度时日志字段保持空值。

Supported inputs include `15`, `十五`, `十五楼`, `按十五`, `button 15`, return
to origin, confirmation, cancellation, and emergency stop. Bare numbers are
accepted only as proposals and require a second utterance such as `确认`.

Dry-run examples (no robot motion):

```bash
python3 run_voice_command_gateway.py --text 15 --text 确认
printf '%s\n' '{"text":"十五","confidence":0.96}' 确认 \
  | python3 run_voice_command_gateway.py
```

Physical execution remains locked unless both flags are present:

```bash
python3 run_voice_command_gateway.py \
  --execute --acknowledge-physical-motion
```

Microphone dry-run (default; never moves the robot):

```bash
python3 run_voice_microphone.py
```

Keep quiet during the first second so the adaptive energy detector can measure
the room noise. Then say, for example, `送外卖到十五楼` once, without a long
pause. Both transcripts must contain the complete `送外卖到<1..16>楼/层`
command; Arabic numerals and traditional `送外賣到十五樓` are supported.
The former `机器人` prefix and bare `十五楼` are rejected. Pinyin here describes
pronunciation, not supported ASR transcript text. Return remains available as
`送外卖返回原点`. A strict whole-command grammar rejects sentences such as
`送外卖到十五楼吃饭`. Stop/cancel need no prefix; legacy confirmation
still requires an already pending addressed command. Text-only gateway behavior
is unchanged. This is a transcript gate, not an acoustic wake-word model or
speaker authentication: overlapping speakers and ASR hallucinations can still
cause mistakes. No persistent wake window is opened.

The same recording is decoded
twice with different prompts/beam settings; motion is proposed only when both
internal results resolve to the same allow-listed command. The model stays
resident in a loopback-only local process to avoid reloading it for every
phrase. Use `--once` for a one-utterance microphone/ASR check, `--minimum-rms`
for the absolute trigger floor, and `--noise-ratio` for the threshold relative
to measured room noise. The older spoken-confirmation flow remains available
through `--require-spoken-confirmation`.

The microphone path requires the same two flags before physical motion is
possible:

```bash
python3 run_voice_microphone.py --execute --acknowledge-physical-motion
```

After startup, provide one transcript per line. A button press or return needs
confirmation within five seconds. `停止`, `急停`, or `stop` bypasses
confirmation and calls the independent safety-stop bridge. ASR text is never
concatenated into a shell command; button values are range checked to 1..16 and
passed as a fixed subprocess argument list.

Voice recognition is not a safety-rated emergency stop: it must wait for the
speaker to finish and for ASR to complete. Keep the physical E-stop available.
The ASR confidence is calculated from Whisper token probabilities and its
no-speech probability; missing confidence data is rejected rather than trusted.

## Guarded origin recovery

Stationary button acquisition uses a six-frame window with a 350 ms sample-age
limit and a 900 ms span limit, accommodating measured 150–230 ms RGB-D inference
latency. Only frames captured after acquisition starts are eligible. Encoder
motion above 0.003 rad restarts the acquisition epoch. The generic window default
remains 180 ms; the existing motion-time second-capture path explicitly uses
350 ms and a 1.0 s span. Its spatial quality thresholds are unchanged.

A combined press/release/return retains the press safety policy until the
scheduled release passes the prepress position and fresh encoders confirm exit
from the 5 cm near-panel zone, including when return visual monitoring remains
enabled. A safety-stopped press does not automatically clear
its latch or send a competing recovery trajectory. A controller response timeout
also inhibits automatic recovery because the original command may still exist.

For an explicitly supervised test, `run_button_batch.py 15
--pause-depth-near-button` suppresses depth alarms only within 5 cm of the live
target, at most 2 cm laterally, with at most 1 cm overtravel. The option is off
by default. A 250 ms lease requires fresh, healthy encoders; the pause lasts at
most 15 seconds and cannot reactivate after retreating beyond 5 cm. Depth
processing continues, and its alarm action resumes when the lease expires or
the arm exits the zone. Vision communication errors still request a stop.
Controller collision validation and the independent stop bridge stay enabled.

`run_clearance_return_to_origin.py --hold-report <report.yaml>` now defaults
to controller-side dry-run validation, without motion or latch reset. Execution
still requires both physical acknowledgment flags. It generates a direct return
and relies on controller-side MoveIt validation of the real `top_chassis_link`
with 5 mm padding and the retained `piper_camera_base_link` collision mesh.
The former 10 cm elbow/TCP height proxy and +/-6 degree low-elbow swing rule
have been removed. A direct path that touches either real mesh is rejected.

Physical recovery uses absolute near-field visual guarding, disables the
depth-degraded allowance, publishes encoder camera poses, and checks preflight.
It does not capture a new background reference. A visual stop stays latched;
this recovery does not automatically retry or reset after the obstruction clears.
Pose-monitor failure and interrupted execution also request an independent stop.
These checks do not replace a physical E-stop or full environmental coverage.
