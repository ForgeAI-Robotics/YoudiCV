# 本地语音部署记录（2026-09-15）

## 当前入口

`run_voice_assistant_gui.py -> run_voice_streaming.py -> CommandDispatcher -> run_button_batch.py -> run_button_light_closed_loop.py`

GUI 当前唯一活动模型是 `sherpa-onnx 1.13.8 + SenseVoiceSmall int8`。Whisper small 与旧流式 Paraformer只保留供人工回滚；运行事件固定记录 `whisper_fallback: false`，不会自动调用其他模型。

## 数据流与命令门

1. `parec` 连续读取 DOV 麦克风的 16 kHz 单声道 s16le，每帧 512 个采样（32 ms）；队列积压时丢弃最旧帧，避免稍后执行过期语音。
2. sherpa-onnx Silero VAD 使用阈值 0.55、最短语音 250 ms、尾静音 350 ms，把完整短句送入常驻 SenseVoiceSmall。
3. 语音开关变化时同时重置 VAD、音频队列和唤醒窗口。
4. 准确识别中文“一二三”后回复“在的”并保持唤醒，直到完整任务开始执行、手动关闭语音或听到“没事了”。任务必须由原始识别文本严格匹配“送外卖到 1–16 楼”；不使用同音词表、句首替换、缺词补全或 Whisper 回退。
5. SenseVoice 的 sherpa-onnx 接口没有可用的校准置信度，所以日志写 `confidence: null`，不虚构分数。停止和取消仍可在未唤醒时触发。

## 为什么替换旧 Paraformer

现场日志证明麦克风和 VAD 正常，但旧模型把“优小妹”识别成“有小妹/幼小妹”，把“送外卖到十五楼”识别成“动物外卖到十五楼/动外卖到十五楼”。用词语替换修正只能覆盖已见错误，已全部撤销。

SenseVoiceSmall 使用完全相同的历史录音和 Silero 分段得到：

- 两段楼层短语：均为“十五楼”，解码 0.088/0.094 秒。
- 完整任务：原样输出“送外卖到十五楼”，解码 0.187 秒，RTF 0.055。
- 模型加载：离线基准 2.70 秒；实际 PulseAudio 安全启动 2.878 秒。
- 官方中文样例：解码 0.225 秒，RTF 0.052。

这些数据只覆盖少量录音，证明新模型在已有样本上更准、更快，不代表 1–16 楼、多说话人的总体准确率。下一步应在桌面安全预览入口采集现场统计，再决定是否允许机器人执行。

## 模型校验值

```text
7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e  sensevoice-int8.tar.bz2
c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51  model.int8.onnx
f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc  tokens.txt
9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6  silero_vad.onnx
```
