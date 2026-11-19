# 机器人按钮系统结构审计（2026-09-14）

## 本次故障链的证据

- `button_light_closed_loop_runs/20260914_150652` 的第 1 次按压在 15.01 mm 处检测到
  -1.185 N 持续法向力；两帧灯态均为 OFF，因此状态机正确发出了 `retry_required`。
- 第 2 次目标刷新和规划完成，实际近距保持为 15.84 cm。随后旧实现再次启动只适用于首次远距接近的
  receding/splice 控制，并因 `matrix_prediction_coverage_insufficient` 进入 `safety_paused`。
- 近距深度周期在第 1 次回退时被标记为 finished，但第 2 次向面板运动前没有重置，因此覆盖降级策略没有重新进入。
- `wait_terminal()` 对 `safety_paused` 不计算超时，导致主程序一直等待；终止本地进程前，OrangePi 仍保留异步轨迹。

## 已修复的结构问题

1. **按压策略按状态分支**：第 1 次使用移动 RGB-D 与 receding/splice；第 2、3 次从当前近距观测位直接执行新目标的 observation-to-prepress 和力控按压。
2. **每次尝试有独立安全周期**：新尝试会重置近距深度租约、退出残留的回退模式，并重新启用向内运动的视觉安全。
3. **距离契约统一**：状态机的名义 15 cm 允许使用刷新后已验证的 13–17 cm 实际保持位，不再因毫米级平面变化中断。
4. **力方向明确**：只有朝面板方向的负法向力能够触发接触；相反方向的惯性/摩擦残差不能触发回退。
5. **生产入口统一**：`run_button_batch.py` 和语音网关均调用 `run_button_light_closed_loop.py`，不再绕到旧固定距离按压脚本。
6. **配置完整传递**：语音网关的 config 和 hold report 会传到批处理及闭环进程；生产入口统一启用 15 cm 深度降级策略。
7. **可观测性**：报告实时写入 `running` 和当前 phase；每个 checkpoint 同时输出 run id，避免追加日志中的旧 traceback 被当成本次结果。
8. **终止清理**：闭环进程把 SIGTERM 转换为受控异常，使 receding/splice 的异常路径有机会取消远端缓存轨迹。
9. **后台线程所有权**：`PiperUdpRobot` 保存并 join UDP 接收线程，关闭后不会让旧线程读取重开的 socket。
10. **路径可移植性**：OpenCV worker 使用当前 Python 解释器与项目相对路径，不再写死账户和快照目录。

## 仍需分阶段消除的技术债

- `run_online_replan_no_contact.py` 有 4556 行，`main()` 有 3842 行；当前闭环仍从这个可执行脚本导入 7 个通用函数。应把目标采集、几何约束、IK 和异步等待拆到库模块，旧脚本只保留 CLI。
- `run_button_light_closed_loop.py` 的 `main()` 有 573 行并包含多层闭包。应把一次任务重构为显式 controller，对状态、轨迹 owner、取消令牌和恢复策略使用结构化对象。
- 5011 等端口仍有调用点写死，恢复脚本还含固定 waiting joints 和 panel X。应全部进入带版本的配置 schema，并在启动时校验。
- 多个 systemd 按钮单元可以同时被启动；远端 bridge 会拒绝部分并发，但视觉 `set_target` 可能先发生竞争。需要跨入口的单任务 lease。
- 通用 `wait_terminal()` 允许安全暂停无限延长，这是实际障碍物场景所需，但必须持续输出暂停心跳并保留明确的人工取消/恢复事务。
- 全项目有 58 个宽泛异常捕获。多数位于安全清理边界，但仍需逐个区分传感器暂态、规划失败、控制器失败和终止请求，避免错误原因被覆盖。

## 验证基线

- 本轮关键链测试：70 passed。
- 修改后完整测试：349 passed（10 个既有的全 NaN 深度窗口 warning）。
- 实机恢复报告：`button_recovery_runs/20260914_151805/recovery_report.yaml`，原位最大关节误差 0.001 rad。
