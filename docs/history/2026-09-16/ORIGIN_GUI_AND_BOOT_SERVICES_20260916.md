# GUI 原位控制与开机服务

## GUI

- `检查原位` 只读取 Piper 编码器，将六关节与记录原位比较。
- `返回原位` 复用 `run_direct_return_to_origin.py` 的 MoveIt 预校验、轨迹桥和安全锁存复位。
- 两个操作与楼层任务共用 `CommandDispatcher`，同一时刻只允许一个机器人操作。
- 两个按钮共用一个状态块，显示是否在原位、最大关节误差或失败原因。
- `中止任务并返回` 先锁存独立安全停止并终止活动任务，等待任务进程退出后才串行启动返程；按钮与检查/返程共用状态块。
- 优小妹 GUI 没有加入任何开机启动项。

## 开机服务

Jetson：

- `button-vision.service` 已启用，负责 Orbbec、YOLO 与 RGB-D 安全服务。
- Clash Verge、Bluetooth 和 SSH 保持各自现有系统服务配置。

OrangePi：

- `piper-ros-bringup.service` 无窗口启动 ROS、Piper 驱动和 MoveIt。
- 5010、5011、5013、5014 四个桥要求 ROS bringup 服务先启动。
- 原 XFCE `start-terminal.desktop` 被停用，避免重复启动 ROS 和弹出终端。
