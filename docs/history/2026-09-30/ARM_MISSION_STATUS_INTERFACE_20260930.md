# 机械臂任务状态接口（schema 1）

本接口把机械臂按钮操作变成可关联、可审计的事件流，供优小妹/Android GUI 和后续导航适配器读取。当前实现只发布机械臂及按钮阶段；它不控制底盘，也不声称导航或整趟电梯任务已经完成。

## 状态通道

- 默认实时 outbox：`mission_status_runs/mission_events_YYYYMMDD.jsonl`，每天一个文件，一行一个 UTF-8 JSON 事件。
- 发起任务的 `CommandDispatcher` 把 outbox 路径、`task_id`、`mission_id`、`segment_id` 经环境变量传过 `run_button_batch.py` 到 `run_button_light_closed_loop.py`。
- 任务进程从深层状态机发布状态；父调度器增补接收、进程退出和分段等待人工操作状态。多个进程通过文件锁追加，`sequence` 在该文件内严格递增。
- 每个事件带 `schema_version`、全局 `sequence`、唯一 `event_id`、任务/使命/分段 ID、墙钟和单调时钟、来源、阶段、结果、原因码及可用证据。`task_id` 关联一次机械臂执行；分段乘梯的呼梯按钮与目标楼层按钮有不同 `task_id`，共享一个 `mission_id`。
- `emitted_monotonic_sec` 只用于同一发送主机上的间隔审计；跨设备先按每设备 `sequence` 去重/排序，再以接收端时间和 `event_id` 关联，不能直接相减不同主机的 monotonic 时间。
- 原有 `button_light_closed_loop_runs/<run>/journal.jsonl` 仍是逐步物理执行的权威审计记录；状态 outbox 是 GUI 和集成方读取的规范化实时镜像。每个深层状态先 fsync 原始 journal，再写状态镜像，镜像不再额外同步刷盘，避免在按压/回撤关键路径增加第二次 fsync 阻塞。

示例：

```json
{"schema_version":1,"sequence":18,"event_id":"…","task_id":"…","mission_id":"…","segment_id":"target_floor","emitted_at_unix_sec":1790760000.1,"emitted_monotonic_sec":12345.6,"producer":"arm_pressing","phase":"press_feedback_received","outcome":"active","attempt":1,"reason_code":null,"evidence":{"attempt":1,"bridge_status":"contact_confirmed","feedback_contact_confirmed":true,"encoder_error_rad":0.004},"raw_event":"press_attempt_finished"}
```

## 阶段语义

常用阶段包括 `accepted`、`arm_task_started`、`preflight`、`approaching_button`、`pressing`、`press_feedback_received`、`observing_press_result`、`retry_required`、`button_on_confirmed`、`waiting_for_floor`、`floor_arrival_observed`、`returning_origin`、`origin_verified`、`arm_button_task_completed` 和 `failed`。重复按压必须由 `retry_required` 表达，证据会带尝试次数、按压桥状态、接触确认、编码器误差和灯态观察结果。

分段呼梯的流程在呼梯按键机械臂操作后发布 `call_button_segment_completed`，然后发布 `awaiting_operator_inside_elevator`；用户/导航程序确认底盘已经进入电梯后，才会开始目标楼层按钮分段。目标按钮分段返回后发布 `target_floor_button_segment_completed`。这只是“按键、等待灯灭、机械臂返原位”的结果；底盘驶入、乘梯导航、驶出并到达最终交付点仍须由导航系统完成并单独发布它自己的任务结果。

`outcome` 采用 `accepted`、`active`、`waiting`、`degraded`、`completed`、`failed`。`reason_code` 是机器可读原因；原始值和数值证据保留在 `evidence`，便于定位灯态不确定、重定位失败、力反馈结果和原位验证失败。

## ROS/导航适配边界

当前 AGX 上 `/usr/bin/python3` 可导入 ROS1 `rospy` 与 `std_msgs`，并能只读访问底盘 Master `192.168.168.100:11311`；ROS Master 环境变量默认未设置，必须由专用服务显式指定。当前 ROS graph 没有 `/mission` 或 `/youxiaomei` 命名空间的话题/服务。导航公司尚未给出最终 topic/message、状态消费规则或任务 ID 约定，所以状态 outbox 目前仍未通过 ROS 发布，也没有向 Android 厂商 UDP 端口发送任务完成包。未来可先在独立 `/youxiaomei/mission/*` 命名空间做影子适配，再由导航公司确认正式的 typed action/service；不得调用 Android APK 的 `active_result` 命令。

本机核对 ROS1/ROS2 Python 客户端后，再按导航方确认的契约添加 ROS adapter；在此之前 outbox 可直接作为接口评审样例和端到端状态源。
