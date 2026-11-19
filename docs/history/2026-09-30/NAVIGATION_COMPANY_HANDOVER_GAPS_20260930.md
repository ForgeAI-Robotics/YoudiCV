# 给底盘导航公司的联调缺项清单（2026-09-30）

## 结论

现有设备可以分别运行底盘导航栈和机械臂电梯按钮任务，但尚不能由 Android/AGX 的一个订单自动完成“起点导航到电梯 → 呼梯 → 进入电梯 → 目标楼层 → 出梯 → 目的地”的端到端任务。底盘 ROS 中存在地图、`move_base`、电梯/地图切换组件和 APK UDP 桥；机械臂一侧有独立按钮闭环。缺少两套控制域之间的正式任务 API、底盘/机械臂控制权交接，以及贯穿两侧的任务状态反馈。

本文件区分：**在线运行图已观测到的接口**、**APK 静态代码中的接口**、**尚未实机验证或需要导航公司定义的内容**。本次没有向 ROS 发布任务、没有调用服务、没有驱动底盘或机械臂。

## 当前部署与在线证据

| 部件 | 当前已知职责与地址 |
|---|---|
| 底盘 Jetson / Ubuntu | `192.168.168.100`，ROS Master `:11311`，ROS Melodic；运行 `move_base`、`multiple_map_switching_node`、`udp_commit_node`、`chassis_server_node`、厂商 `velocity_smoother`。 |
| Android 控制屏 | `192.168.168.110`；RobotArm APK 的 ROS/UDP 客户端连底盘 `192.168.168.100:8888`。 |
| AGX/上位任务管理机 | 已核实当前工作机 hostname `ubuntu` 是 NVIDIA Jetson AGX Orin Developer Kit；以太网主地址 `192.168.168.130`，`.162` 是同一接口兼容别名。建议 AGX 任务管理器运行在 `.130`；Android APK 在 AGX 模式下仍从 `.110:5005/UDP` 发到 `.162:5005/UDP`，保持该兼容入口。项目笔记中的 Android 状态入口 `UDP/5002` 之实际绑定设备、方向和报文格式仍待厂商确认。 |
| 机械臂控制机 | OrangePi `192.168.168.120`，独立 ROS Master `:11311`，运行 Piper/MoveIt/轨迹与按压桥。 |
| 上位视觉/任务机 | 项目交接基线记为 `192.168.168.130`；`.162` 曾作为此上位机的别名/接口使用，部署时需确认实际网卡和地址映射。 |

当前底盘 ROS Master 在线图确认：

- `/map` (`nav_msgs/OccupancyGrid`) 由 `/multiple_map_switching_node` 发布；`/static_map` 由同一节点提供。
- 地图参数指向 `/home/ubuntu/std_wr_demo/map`，地图 ID/文件参数为 `jxb_105.yaml`、`jxb_105.pgm`。Android ROS 节点没有发布或订阅 `/map`。地图由底盘 Ubuntu/Jetson 栈加载并向 ROS 发布；没有证据显示 Android 上传地图文件。
- `/udp_commit_node` 发布 `/UD/elevator/send_goal` (`custom_msgs_srvs/ElevatorSendGoal`)，由 `/move_base` 与 `/multiple_map_switching_node` 订阅。
- `/SWJ/ApkReceiveResult` (`custom_msgs_srvs/ApkReceiveResult`) 由多个底盘节点发布并由 `/udp_commit_node` 订阅；`/SWJ/ApkSendWork` (`custom_msgs_srvs/ApkSendWork`) 有底盘节点和 `/udp_commit_node` 发布者。这说明厂商 ROS/UDP 桥具备任务/结果消息通路，但不等于本项目 Android GUI 已用它完成往返。
- 当前开发机缺少 `custom_msgs_srvs` 包及 `.msg/.srv` 定义，无法读取 `ElevatorSendGoal`、`ApkReceiveResult`、`ApkSendWork` 等字段。请导航公司提供与运行版本完全匹配的消息包、服务定义、字段语义和版本号。

## 本项目已具备的机械臂侧能力

- 接收按钮目标（楼层按钮 1–16，或呼梯用上/下按钮），调用生产按钮闭环：按钮定位、MoveIt/IK 与轨迹检查、接触反馈按压、灯态 ON/OFF 判断、重试及机械臂返原位。
- 独立状态至少包括任务启动、按压/重试、按钮 ON/OFF/不确定、安全暂停、任务成功/失败和回原位结果；完整状态定义与数据来源见 `CHASSIS_NAVIGATION_ARM_BUTTON_HANDOVER_20260916.txt` 第 3、5 节。
- 项目 GUI 当前监听 Android `.110 -> .162:5005` 的 UDP 包。只将 `bt_control command=10` 解释成楼层目标；`command=1,value=arrive` 和 `command=5` 地图 ID 只显示/记录；其他命令不执行。
- GUI 的“完整分段乘梯”目前只会根据手工设置的当前楼层选择上/下按钮并执行机械臂按压；呼梯任务结束后进入 `waiting_square`，等待人手将底盘驶入电梯并按 GUI“继续”，之后执行目标楼层按钮任务。这不是完整自动乘梯，也没有目标楼层出梯后的路线。
- GUI 对 Android 的 `accepted`/`null` 是传输层应答，不是任务完成回报。任务完成/失败目前只更新 GUI/本地日志，没有将最终状态和原因发回 Android/AGX。
- APK 静态代码存在 AGX `active_result` 输入、底盘 ROS UDP 初始化/目标命令及结果监听；APK 的结果码 `60` 分支会向 `.162:5005` 发送 `bt_control command=2`。这说明厂商方案有预留回传，但当前现场记录没有完整的 AGX→Android→Jetson→机械臂→Android→AGX任务样本。
- **不要把 `active_result` 当作通用任务状态回报来试发。**当前项目约束明确禁止网关发送它：APK 的若干 command（包括 2/4/6/7）会进入底盘位姿、地图初始化、暂停/恢复或厂商 Binder 硬件分支。公司需定义与动作命令隔离的任务状态消息/ACK，或提供每个 command 的完整安全语义与联锁后再联调。

## 请导航公司明确并提供的项目

### 1. 任务级接口与订单字段

请提供正式、带版本的任务接口（推荐 ROS action 或等价的可靠 RPC，而不是只发无状态 topic），并说明由 AGX、Android 还是 Jetson 接收订单。至少需要：

- 全局唯一 `task_id`、幂等规则、重复包处理、请求确认/超时与取消方式；当前 Android 测试包只有 sequence，GUI 没用它作为任务 ID。
- 起点/当前位置、目标楼层、地图/楼栋/电梯 ID、目的地位置或 waypoint、订单类型。当前项目 GUI 仅接楼层号；Android `command=5` 地图标识只被记录，没有起点/终点导航字段。
- 明确一条订单如何映射为各导航子目标：起点 → 电梯前呼梯工作位 → 电梯内站位 → 目标楼层电梯内站位 → 出梯点 → 最终目的地。
- 每个阶段的目标坐标系（`map`、`odom`、`base_link` 等）、坐标单位、朝向约定、容差、地图切换规则及失败重试策略。

### 2. 导航与按键工作位

- 提供电梯呼梯面板和轿厢楼层面板的 waypoint/marker、面板 ID、所在 map/floor、可接近方向和面板朝向；若一个电梯有多套面板，要能区分。
- 和机械臂方共同标定底盘最终工作位：`map -> base_footprint` 目标位姿、位置/偏航容差、到位后稳定时间、局部定位质量门限。该工作位必须位于已验证的相机视野和机械臂可达范围内；导航公司需能保证到达并反馈是否满足容差。
- 明确电梯门开/关、轿厢到达、进梯许可、出梯许可、乘梯中、目标楼层确认等权威状态来自哪个节点/传感器、字段含义和失效行为。不能仅以“按钮灯灭”代替轿厢门可通行或已安全出梯确认。
- 说明当前 `/UD/elevator/send_goal` 和 `/UD/elevator_initial/auto_match_map_srv` 的生产用途、调用方、字段、成功/失败码、对 map 与 floor 的影响；明确这些接口是否支持完整多楼层导航，还是只负责单个电梯/地图匹配阶段。

### 3. 控制权与安全交接

- 提供一个唯一的 chassis command arbiter/mux 或 lease API，明确 move_base、手柄、AGX 与机械臂任务期间谁拥有底盘运动权；禁止多个发布者同时争用 `/cmd_vel`。
- 进入按键阶段前需有可读回执：取消/暂停导航、底盘零速度、里程计线/角速度低于双方约定阈值、驻车/制动状态有效、无急停/碰撞/门区故障；该确认必须带时间戳和有效期。
- 按键期间保持底盘锁定；机械臂报告已返原位且编码器验证通过后，才释放导航租约。租约/心跳超时、网络中断、任一侧进程崩溃或急停时，底盘 fail-safe 停车并报告 fault；说明恢复需人工确认还是可自动恢复。
- 提供只读的底盘状态接口：`/odom`、速度、驻车/电刹、急停、碰撞条/bumper、门状态、导航状态、定位质量、楼层/地图 ID、控制模式、故障码及传感器健康；给出 freshness/timeout 定义。
- 物理急停保持最高优先级。按钮任务的视觉暂停、按压失败、机械臂异常和人工中止如何通知导航栈，需有明确联锁和恢复规则。

### 4. 任务状态反馈、告警与回传

请统一 Jetson ↔ AGX/Android ↔ 机械臂上位机的状态契约。至少要有 `task_id`、阶段、时间戳、百分比/子目标、结果码、可读说明和是否可重试。建议状态：

`ACCEPTED / REJECTED_BUSY / NAV_TO_CALL_PANEL / CALL_PANEL_ARRIVED / BASE_HOLD_CONFIRMED / ARM_TASK_REQUESTED / CALL_BUTTON_CONFIRMED / WAITING_FOR_BOARDING / BOARDING_CONFIRMED / NAV_IN_ELEVATOR / TARGET_FLOOR_REACHED / FLOOR_BUTTON_CONFIRMED / WAITING_FOR_ELEVATOR_DOOR / EXITING_ELEVATOR / NAV_TO_DESTINATION / COMPLETED`，以及 `PAUSED_OBSTACLE / ABORTING / ABORTED / FAILED`（附结构化 `reason_code`）。

- 明确哪些状态由导航公司负责、哪些由机械臂系统负责，以及哪个节点汇总并保证状态顺序。
- 任务接受、阶段变化、最终成功/失败必须有应用层 ACK；区分 UDP 收包 ACK、任务已接受 ACK、任务完成结果。定义重发、去重、乱序、断线缓存/补发和状态查询。
- 当前本项目 GUI 对 `command=2` 不作任务状态处理，也不主动向 Android 回报完成状态。虽然 APK 有 `command=2` 的结果回传分支，但不能直接复用为本项目 ACK；需导航公司说明它的状态码、方向、动作副作用和安全联锁，优先定义不复用动作命令的新状态接口。
- 当前日志未记录到真实 `active_result`、`ApkReceiveResult` 结果回传闭环。请导航公司提供其 UDP commit 节点协议、Android/AGX 接收端日志和一个已完成任务的脱敏示例包。

### 5. 厂商控制接口与已知风险

- 请交付当前运行版本的 ROS 包/消息定义/launch/参数说明，特别是 `custom_msgs_srvs`、`udp_commit_node`、`multiple_map_switching_node`、`move_base` 与 chassis command arbiter 的版本/哈希及部署方式。厂商源代码不要由本项目代改。
- `cmd_vel` 当前链路是 `/cmd_vel -> velocity_smoother -> /plt_cmd_vel -> chassis_server_node -> /dev/udbot -> 轮电机`。已有手柄实测中，输入 `+0.25 m/s` 时 `/plt_cmd_vel` 仅约 `+0.012–0.015 m/s`，`-0.25 m/s` 曾被限为 0。请解释厂商 smoother 的限速策略、导航与手动控制是否共用此路径、最大速度/加速度/角速度/制动距离，以及是否有配置会影响自动导航。不要为了本项目绕过厂商安全平滑器。
- 明确现有自动导航完成后 move_base 是否保证停止发布、停车制动是否保持、任务失败/取消后如何清零速度和清障恢复。
- 说明定位重置/地图切换失败、楼层识别不一致、电梯被占用、门未开、无法进出梯时导航任务如何终止并上报。

## 双方接口责任建议

| 责任方 | 提供/负责 |
|---|---|
| 导航公司 | 地图与多楼层定位；起点/电梯/出梯/最终目的地导航；电梯/门/楼层权威状态；底盘停车、控制权仲裁、急停/故障接口；ROS/UDP 消息定义；导航阶段状态和故障码；底盘工作位坐标及到位容差。 |
| 机械臂项目 | 按钮定位、按压和灯态判定；机械臂状态/健康/原位验证；按键成功/失败、重试、视觉暂停和返原位结果；提供所需电梯面板工作位可达边界。 |
| 双方共同 | task ID 与全程状态机；按键阶段 chassis lease；楼层/面板/电梯 ID 映射；门与按钮判据的顺序；网络中断/急停/取消处理；端到端日志与验收。 |

## 建议验收顺序

1. **接口冻结（只读）**：交换 msg/srv/action 定义、状态码、topic/service/action 名、端口、帧样例、ROS 版本和设备地址；用抓包与只读 ROS 图确认每个端点。
2. **仿真/空任务**：同一 task_id 走全状态，不发布速度/机械臂运动；验证状态单调、有 ACK、重复消息幂等、断链可恢复。
3. **底盘独立导航**：机械臂收回并锁止，验证从起点到呼梯位、进梯位、出梯位、最终目的地位的定位误差、速度曲线、停止、地图切换及故障处理。
4. **固定面板位联锁**：底盘低速到标定工作位，确认零速与 lease，再由机械臂只做视觉识别/无接触规划；确认后才单按钮按压。
5. **完整乘梯分段**：呼梯、门开、进入、目标楼层按钮、灯态/楼层到达、门开、驶出、最终导航；每阶段记录两侧时钟同步日志和同一 task_id。
6. **异常验收**：障碍物、定位丢失、门不开、按钮不亮、机械臂未回原位、AGX/Android/ROS 断链、急停、取消和节点重启；检查底盘停稳、权限不误交接、错误原因明确。

## 在线证据与代码定位

- 在线 ROS graph/参数只读查询：`192.168.168.100:11311`；地图与任务接口结果另记于 `AGX_MAP_MISSION_FLOW_20260930.md`。
- 机械臂对接基线、网络端口、状态和验收流程：`CHASSIS_NAVIGATION_ARM_BUTTON_HANDOVER_20260916.txt`，尤其第 3、5、6、10–12 节。
- Android UDP 协议：`piper_handeye/android_agx_protocol.py`；当前网关白名单与只回传传输 ACK 的行为：`run_android_command_gui.py`；任务分段 `start_trip()/resume_trip()`：`run_voice_command_gateway.py`。
- 最近本地 Android UDP 记录为 `android_command_gui_runs/android_commands.jsonl`，只有楼层 `command=10` 样本；没有真实任务完成状态回传样本。
- 机械臂按钮流程现已提供 schema 1 实时状态 outbox：`mission_status_runs/mission_events_YYYYMMDD.jsonl`；字段、阶段定义、任务 ID 关联和 ROS adapter 边界见 `ARM_MISSION_STATUS_INTERFACE_20260930.md`。AGX 汇聚状态并按阶段签发机械臂/底盘绿灯的目标状态机与仿真契约见 `AGX_MISSION_MANAGER_DESIGN_20260930.md`。两者目前都没有部署为跨设备传输服务，不能表示 AGX 已开始实际授权硬件。
