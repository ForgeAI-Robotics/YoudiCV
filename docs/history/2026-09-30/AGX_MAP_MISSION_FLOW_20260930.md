# 地图、AGX 任务与任务状态回传调查（2026-09-30）

本次只读检查：未调用导航/地图切换/电梯服务，未发送任务或运动命令。

## AGX 管理器部署主机身份

- 当前工作主机 `hostname=ubuntu`，设备树型号为 NVIDIA Jetson AGX Orin Developer Kit，JetPack R36.4.7/aarch64；以太网主地址为 `192.168.168.130`，`192.168.168.162` 为同一网卡上的兼容别名。
- 因此计划中的中央任务管理器部署目标是 AGX `.130`；底盘 ROS 主机 `.100`、Android 屏 `.110` 与 OrangePi `.120` 是独立设备。`AGX` 与 `Android` 不应再在接口文档中混称为一台设备。

## 地图当前由谁持有和发布

- 当前 ROS Master 是 `192.168.168.100:11311`。在线图中 `/map`（`nav_msgs/OccupancyGrid`）由 `/multiple_map_switching_node` 发布；`/static_map` 服务也由该节点提供。
- 在线 ROS 参数为 `/multiple_map_switching_node/UD_MapPath=/home/ubuntu/std_wr_demo/map`、`UD_map_yaml=jxb_105.yaml`、`UD_map_pgm=jxb_105.pgm`；`/multi_sensor_fusion_localization` 也有同一地图路径参数。运行时地图源因此在底盘 Ubuntu/Jetson 的文件系统/ROS 栈一侧。参数只能证明配置路径，不能排除该路径本身是软链接或网络挂载。
- Android 当前 ROS 节点 `/android/remotemanager` 和其 `/listener` 节点均未发布或订阅 `/map`；Android 上有若干状态/传感器 listener，但没有地图栅格流。因此目前证据不支持“导航地图由 Android 发布或存放”。
- APK 的 AGX 命令可以携带地图/点位标识并请求初始化/匹配，但这是发地图 ID/位姿请求，不是把整张地图从 Android 上传。当前项目 GUI 收到 `command=5` 的地图标识时只显示并记录，不触发底盘地图初始化。

## AGX/Android 到 Ubuntu 的任务路径

- Jetson 的 `/udp_commit_node` 发布 `/UD/elevator/send_goal`，该话题由 `/move_base` 和 `/multiple_map_switching_node` 订阅；这是当前在线图上实际连接的导航目标入口。
- RobotArm APK 的 ROS UDP 客户端固定连接 `192.168.168.100:8888`，并包含 `SendGoal`、`NewSendLiftInitial` 等 ROS 封装。APK 的 AGX `active_result` 监听器可解析外部命令；其中地图/点位初始化分支会调用 `NewSendLiftInitial`。因此厂商应用设计上存在 Android 作为 AGX 与 Jetson ROS 之间网关的命令路径。
- 当前项目另有 `/home/qianjue/contral_test/run_android_command_gui.py`：它在 `192.168.168.162:5005` 只接受源地址 `192.168.168.110` 的 Android UDP 包。白名单功能目前只把 `bt_control` 的 `command=10` 楼层号映射为机械臂按钮任务；`command=1,value=arrive` 只记录，`command=5` 地图 ID 只记录。楼层任务执行需要 GUI 双重启用；完整分段模式还需当前楼层，并在呼梯按键后由操作员点击“已进入电梯，继续按目标楼层”。
- 本地 `android_commands.jsonl` 只记录到 Android 发来的 `bt_control command=10`（目标 5、8 楼）重复包；没有 `active_result`、地图初始化或真实导航任务的端到端样本。

## Ubuntu/ROS 到 Android/AGX 的反馈路径

- 当前 Jetson ROS 图中 `/SWJ/ApkReceiveResult` 由多个厂商任务节点发布，`/udp_commit_node` 订阅；`/SWJ/ApkSendWork` 也由 `/udp_commit_node` 和任务节点发布。这证实 ROS 与厂商 UDP/Android 桥存在双向任务相关消息通道。
- APK 的 `UdpServer` 识别 ROS 转发的 `SWJ/ApkReceiveResult` 并交给 `EventResultListener`。AGX 模式下，RobotArm APK 对结果码 `60` 有明确分支：通过 UDP 向 `.162:5005` 回发 `bt_control command=2` 和结果文本。这是已在代码中找到的任务结果回传机制。
- 当前自建 GUI 的 `AndroidUdpListener.reply()` 仅回 `accepted` 传输 ACK 或 `null` 停止重发；任务执行完成/失败只更新 GUI 和本地日志，代码没有把完成状态发回 Android/AGX。GUI 收到 `command=2` 也只会按未知命令记录，不会作为任务反馈状态展示。
- 因而，厂商 APK + Jetson 的 ROS UDP bridge 具有设计上的任务/结果通道；但本地新 GUI 的楼层任务链尚未实现任务完成状态回传。现有日志也没有证明厂商 APK 的 AGX 回传分支在当前设备上真实跑通过。

## 判断

1. 当前导航地图在底盘 Jetson 的 ROS/本机地图路径上，不在 Android ROS 节点上发布；Android 至多携带地图/点位 ID 发起切图或初始化请求。
2. “AGX 下任务 -> Android 网关 -> Jetson ROS”在厂商 APK 和 Jetson 当前 ROS graph 中有代码与接口依据；但唯一现场 UDP 记录是楼层 `command=10` 到 `.162:5005`，不能据此宣称真实 AGX 导航任务已端到端验收。
3. 厂商 ROS/UDP 栈具备结果码向 Android 转发、APK 再用 `command=2` 向 `.162:5005` 回传的实现。自建 GUI 目前只回传 transport ACK，不回传 mission completed/failed，所以 AGX 若通过该 GUI 发起任务，当前无法收到可信的任务完成状态。
4. 下一步若要完成验证，需要抓取一轮不执行运动的 AGX/Android 测试包，确认命令 ID、任务 ID 与应答格式；再让自建 GUI 对任务生命周期发送明确的 accepted/running/completed/failed 消息，并以 AGX 收到的 ACK/状态和 ROS 日志对齐验收。真正的导航任务执行应另行授权并在安全场地测试。
