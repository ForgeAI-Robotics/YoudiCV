# Whisper base / tiny 已有录音对比

测试使用项目下载脚本提供的多语言ggml-base.bin与ggml-tiny.bin，保持现有CUDA、4线程、4kHz滤波、提示词及双遍复核设置。没有改变生产默认small模型。没有打开麦克风、连接机械臂或发送任何动作。

输入与上一轮small对比一致：三份已有录音得到五个VAD片段，一个无VAD片段的整段录音，加两秒合成静音。每个输入重复两次。small数据来自上一轮，base/tiny来自本轮；系统负载不完全相同，样本很少，不能声称总体准确率。

## “送外卖到十五楼”4.34秒片段

| 模型 | 单遍中位 | 双遍中位 | 两次复核结果 |
|---|---:|---:|---|
| small | 1.105秒 | 2.185秒 | 都通过，15号 |
| base | 0.477秒 | 0.969秒 | 都通过，15号 |
| tiny | 0.357秒 | 0.743秒 | 都通过，15号 |

这张表只描述该片段，不代表全语料平均或最坏耗时，也不包括录音、句尾等待和模型首次加载。

## 短指令和负对照

在small原本复核通过的三个15号相关片段中，base和tiny仅完整“送外卖到十五楼”通过，其余两段复核拒绝，两轮一致。base有一次短语主识别为“鞋儿五楼”，纯解析得到5号，但复核阻止了该命令；tiny将另一段输出为“这无聊”。

base直接识别合成静音时产生大量重复“鸿”，置信度约0.945，双遍耗时6.10–6.22秒，最终拒绝。tiny在噪声片段产生重复文本，双遍约4.6–4.8秒，最终拒绝。这说明小模型并非所有输入都更快，高置信度也不能单独证明命令正确。静音/无VAD整段是绕过实时VAD的负对照，并非发生了实际误动作。

结论：base/tiny值得作为速度实验选项，但当前录音已显示短命令识别退步，因此保留small及双遍复核默认设置。后续若尝试base优先、失败回退small，需要计入失败时两套模型的累计耗时，并覆盖更多楼层与说话人再评估。

原始报告：optimization_runs/20260908_voice/base_comparison.json、tiny_comparison.json、mode_comparison.json；对应whisper.log记录GPU后端。模型校验值在downloaded_models.sha256。

可复现命令（仅读录音）：
```
/home/qianjue/anaconda3/envs/piper-handeye/bin/python tools/benchmark_voice_modes.py --model base --output optimization_runs/20260908_voice/base_repeat.json
/home/qianjue/anaconda3/envs/piper-handeye/bin/python tools/benchmark_voice_modes.py --model tiny --output optimization_runs/20260908_voice/tiny_repeat.json
```
本轮仅给基准脚本增加模型/输出/重复次数参数，已语法检查并完成两种模型各14组实际测试；识别服务均在测试后关闭。
