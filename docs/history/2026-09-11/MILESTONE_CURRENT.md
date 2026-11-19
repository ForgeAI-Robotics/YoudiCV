# 2026-09-11 17:22：滚动电流限值加100 ms持续载荷，15号复测成功

最终报告：`button_light_closed_loop_runs/20260911_172204/closed_loop_report.yaml`。

- 状态 `arrived_after_verified_press`，一次按压，完整 OFF→ON→OFF，返回原点，
  视觉安全事件为空。推进速度恢复并保持 10 mm/s。
- 根因是旧算法在深度推算的 10 mm 行程处冻结电流参考；实际接触位置因深度
  误差改变后，后续姿态/加速度电流被误算成载荷。之后试验的跨轴 L2 固定阈值
  虽能按亮，但在 19.858 mm、0.398 A 才停止，用户观察到按压过硬；该算法已
  从生产配置和协议删除。
- 新算法用撞击前 40 个 200 Hz 样本拟合每轴线性运动电流趋势及 MAD 噪声，
  每 5 ms 重算限值。当前样本不参与自身阈值；首次越界后冻结即时参考，J5
  至少5帧且连续越界100 ms才确认 `sustained_rolling_current_limit`。
- 25 ms版本的重复测试 `20260911_171801` 在6.135 mm被短撞击尖峰触发，按钮
  未亮并正确回原点。100 ms最终版在7.960 mm触发，连续载荷104.69 ms、22个
  反馈样本，J5相对撞击前预测值增加0.192 A；按钮成功点亮。22 mm仅为最大
  安全边界。
- 所有按钮和每次重试都必须通过电流反馈桥。只有
  `contact_retracted + feedback_contact_confirmed=true` 才能进入灯态观察；
  桥缺失、硬停、反馈过期、超时或最大行程结束都不能冒充按压完成。
- 六轴绝对电流及滚动相对硬限制始终有效。反馈日志保存停止前最后 40 帧，
  包括预测残差、行程、持续时间、计数、决策和原因。完整非视觉回归324项通过、
  1 项因基础解释器缺少 OpenCV 跳过。

# 最新验证：2026-09-11 15:36，10号自适应近距安全闭环成功

最终实机报告：`button_light_closed_loop_runs/20260911_153647/closed_loop_report.yaml`。

- 状态 `arrived_after_verified_press`，一次按压，返回原点，视觉安全事件为空。
- 初始为 OFF；按压后15 cm观察位连续两张新帧均为 ON（中心环对比度
  86.24/84.76）；回到初始观察位后为 OFF，因此有完整 OFF→ON→OFF证据。
- 最终按压在16.929 mm、关节5持续电流载荷满足判据后撤回。此前未点亮的
  `20260911_153500` 在12.770 mm、J5残差0.320 A时过早停止；上次成功样本
  为16.677 mm、0.3855 A。据此把J5持续接触阈值从0.25 A校准为0.35 A，
  0.40 A即时硬限制和22 mm最大行程保持不变。
- 两次早期保护停止 `20260911_153036`/`153301` 均未按压，并通过专用恢复
  脚本回原点。第一处根因是RGB配准在正常深度帧阻塞位姿发布；现仅在覆盖
  不足且降级已布防时运行。第二处根因是外部监督器拒绝第一帧覆盖确认状态；
  现只允许 `armed + transient + unavailable_streak=1` 这一帧，第二帧必须
  正式激活，其他原因仍停止。
- 最终相关测试65项通过，项目非视觉服务测试310项通过；视觉服务active。
- 最终快照：
  `version_snapshots/adaptive_close_range_safety_button10_verified_20260911_153800.zip`。

# 待实机验证：2026-09-11 自适应近距视觉安全

- 相机已实机识别为 Orbbec DaBai DCW2，VID/PID `2BC5:06A0`，固件
  `RD44606`；官方密集深度工作距离从 0.20 m 开始。
- 近距覆盖例外现在分为“编码器范围内布防”和“连续两帧覆盖失败后激活”；
  到达布防距离不会关闭深度检测。
- 修复低覆盖时 `blocked_raw` 抹除剩余有效像素前景的漏洞。低覆盖采用独立
  300 半分辨率像素大物体阈值，历史边缘碎片 48--82 像素不会触发。
- 深度覆盖不足时增加相邻 RGB 帧仿射配准入侵检测：去除整体曝光变化，
  手掌级连通区域须连续两帧才停止。按钮亮起的小区域低于该阈值。
- 回退不再在面板前立即关闭覆盖例外；实时深度连续三帧恢复、ready 且未
  blocked 后才恢复完整策略，避免 1 cm 处因覆盖锁存卡住返回。
- 相关基础测试 62 项通过，完整非视觉服务测试 307 项通过；RGB 三种合成
  场景使用视觉环境直接验证（相机移动、整体增亮不触发，新大块前景触发）。
  基础 Python 没有 cv2，因此 RGB pytest 在该环境跳过。视觉服务已重启并
  active；尚未执行机械臂运动或人员入侵实测。
- 快照：`version_snapshots/adaptive_close_range_safety_20260911_154000.zip`。

# 最新验证：2026-09-11 12:56，10 号按钮电流闭环实机成功

最新实机报告：`button_light_closed_loop_runs/20260911_125514/closed_loop_report.yaml`。

- 最终状态 `arrived_after_verified_press`，仅按压一次，返回原点，无视觉安全事件。
- 按压前 OpenCV 判定 OFF；按压后 15 cm 观测位两张新帧均判定 ON；随后持续观测约 24 秒，按钮变回 OFF 后返回原点。这次有明确 OFF→ON→OFF 视觉证据，可确认按钮确实触发。
- 电流闭环在前进 16.677 mm 时检测到持续载荷并立即撤回。旧错误分别在 2.67、5.19、8.15、8.40 mm 由运动电流误触发，均未到达约 10 mm 的名义面板位置。
- 根因修复：接触判断在名义面板位置 10 mm 才启用；使用启用前 40 个 200 Hz 电流样本估计局部运动中位数与 MAD 噪声；接触须连续 10 个样本超过动态包络。22 mm 仅是最大几何行程，不是固定按压距离。绝对电流、反馈时效、超时和最大行程保护全程有效。
- 视觉误停根因是固定面板左边界在运动视角中的新显露区域超过旧 5 个半分辨率像素的重投影邻域。现在使用 10 像素邻域匹配；真实更近障碍物回归测试仍会触发。单独 15 cm 往返实机报告 `button_light_closed_loop_runs/20260911_125112/closed_loop_report.yaml` 通过，原故障区域连续 91 帧 clear。
- 相关测试：按压/闭环 45 项通过，深度安全 48 项通过；全 NaN 深度仍有 3 条预期警告。
- OrangePi 用户服务 `piper-guarded-press-bridge.service` 已部署动态基线版本，UDP 5014。

复现命令（会移动真实机械臂）：

```bash
cd /home/qianjue/contral_test
/home/qianjue/anaconda3/envs/piper-handeye/bin/python \
  run_button_light_closed_loop.py 10 --execute \
  --acknowledge-clear-workspace --pause-depth-near-button \
  --max-press-attempts 1
```

以下为此前记录。

# 最新验证：2026-09-08 13:17

第二轮观测传输与RPC优化已部署。最新实机报告 `online_replan_no_contact_runs/20260908_131736/online_replan_report.yaml`：15号按压轨迹完成并回原点，误差0.002 rad，无保护停止，电气触发未独立验证。运动中采样与规划1.000秒，采样0.761秒，两段静止准备合计3.617秒。详见 `OPTIMIZATION_CAPTURE_20260908.md`。仍使用显式近按钮深度暂停策略。

以下为此前记录。

# 最新优化验证：2026-09-08 11:39

四项顺序优化已部署并实机通过。最新报告 `online_replan_no_contact_runs/20260908_113916/online_replan_report.yaml`：15号按钮轨迹完成并回原点，最大误差0.002 rad，无保护停止，未独立验证电气触发。近按钮深度暂停仍需显式参数。

两段准备等待7.784 → 3.974秒；详见 `OPTIMIZATION_RESULT_20260908.md`，含未改善的运动中采样延迟。原冻结快照保留为回滚基线；以下为此前记录。

# 当前已验证状态：15 号按钮近距深度暂停

记录日期：2026-09-08（Asia/Shanghai）。用户要求记住配置与策略，并保存快照。

## 成功记录

- 实机报告：`online_replan_no_contact_runs/20260908_105313/online_replan_report.yaml`，摘要为同目录 `verification_summary.yaml`。
- 状态 `complete_online_press_returned_origin`，原点最大关节误差 0.001 rad，无保护停止。未独立测量按钮电气触发。
- 最近完整验证：213 项 pytest + 4 项视觉服务 unittest，共 217 项通过；全 NaN 深度测试有 3 条既有警告。
- 快照：`/home/qianjue/snapshots/20260908_110636_button15_near_depth_pause`，对应同名 `.tar.gz`。
- 耗时分析：`PROCESSING_TIME_20260908.md`，以本次成功报告为依据。

## 已验证策略与参数

1. 最新录音 `voice_diagnostics/20260907_180650_nbq3rilo/mono_16k.wav` 双次识别均为“送外卖到十五楼。”，置信度约 0.894 / 0.899。沿用 4 kHz 低通、Whisper small 本地模型和命令一致性检查。
2. 实时 RGB-D 和编码器作为目标依据，抬升时锁定基座偏航，本次抬升 0.24 m。沿用有界姿态 IK、控制器 MoveIt 碰撞与固定障碍物校验。
3. 静止采样需要 6 帧：年龄 ≤0.35 s、跨度 ≤0.90 s、位置 RMS ≤1.5 mm、跳变 ≤3 mm、法向 RMS ≤1.5°；编码器偏移 >0.003 rad 则丢弃窗口重采。只接受开始采集后的帧。
4. 运动中第二次采样使用原有专门窗口：6 帧、年龄 ≤0.35 s、跨度 ≤1.0 s、位置 RMS ≤3.5 mm、跳变 ≤7 mm。通用窗口默认 0.18 s 不等于该专门窗口的实际参数。
5. 按压前距离 0.010 m，独立按压行程 0.015 m，速度参数 0.010 m/s；按压、释放和返回组成连续轨迹。接近速度参数 0.15 m/s，近距参数 0.10 m/s，原有降级策略乘 0.5，返回参数 0.25 m/s。
6. 显式实验开关 `--pause-depth-near-button`：前后距离 -0.01..0.05 m、横向偏差 ≤0.02 m、硬件正常且编码器年龄 ≤0.25 s 时暂停深度告警，最长 15 s。退出 0.05 m 范围后不能再次启用。独立急停、控制器碰撞检查、视觉通信错误停止仍有效，深度处理继续运行。
7. 本次实际在 0.04797 m 处暂停，在 0.05097 m 处恢复，持续 6.262 s。开关默认关闭，本次显式开启，不代表其他按钮或环境已验证。
8. 按压/释放阶段不提前切换返回保护策略；等待释放后，编码器确认退出 5 cm 区域。保护停止不自动清锁存并发送竞争恢复轨迹，控制器响应超时也不自动发恢复运动。

## 复现已验证的物理测试

以下命令会移动真实机械臂，是本次参数记录，不是要求未来会话自动执行：

```bash
cd /home/qianjue/contral_test
/home/qianjue/anaconda3/envs/piper-handeye/bin/python run_button_batch.py 15 --pause-depth-near-button
```

此命令只运行按键阶段；已有音频曾先单独完成双重识别，该命令不重新识别音频。复用相机、标定、模型和环境，不复用旧按钮目标位姿。

## 环境与状态限制

- Jetson 地址 `192.168.168.130`，兼容别名 `.162`；OrangePi `.120`。
- 控制 Python：`piper-handeye`；视觉 Python：`elevator-yolo26`；默认 Python 运行 pytest 但没有 cv2，因此视觉服务 4 项测试在控制环境用 unittest 执行。
- 最近已确认回到原点。未来执行仍需读取实时编码器，快照不代表未来物理状态。
- 直接返回脚本不使用视觉监控。抬升恢复路径曾因无法再抬高 10 cm 而失败，不能认为任意姿态下都可用。
- `VisualSafetySupervisor.preflight()` 内部会重置锁存，不是只读接口。恢复前须确认不存在会因清锁存恢复的挂起轨迹；本次处理该问题时先重启了定时轨迹桥。
- 旧 2026-09-02 的 1/5/16 回归是历史结果，不代表新开关已在全部按钮验证。
## 2026-09-11 16:20 — Button 10 receding IK + 15 cm RGB-only close range verified

- Verified report: `button_light_closed_loop_runs/20260911_161945/closed_loop_report.yaml`.
- Result: `arrived_after_verified_press`, one attempt, OFF -> ON -> OFF, returned to origin, no visual safety events.
- Reaching now uses moving RGB-D (four timestamp-synchronised frames), a dynamic ~9 cm guard solved by `FixedRollFastIkPlanner`, and one C2 splice to the refreshed prepress IK. The 15 cm pose is used only after press.
- Timing: guard IK 248.7 ms, moving-target IK 116.4 ms; handoff at 3.812 s and replacement duration 4.743 s. Continuity errors were all zero.
- Guarded press uses J5 for sustained contact confirmation while hard relative and absolute limits remain active on all joints. Verified travel 18.14 mm and J5 residual 0.418 A.
- Inside 15 cm, unreliable moving depth residuals are suppressed and affine-registered RGB intrusion monitoring remains active through the press and retreat. Verified RGB registration score ~0.9999 with zero changed component pixels.
- The near-mode lifecycle is closed permanently at the 15 cm observation endpoint so encoder noise cannot re-arm it during the return to the waiting/origin pose.
- Validation: 315 passed, 1 skipped (base Python has no OpenCV); direct OpenCV validation passed for camera motion, exposure change, and a hand-sized injected foreground.
