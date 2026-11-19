# Xbox 类手柄整机控制可行性审计

日期：2026-09-15

## 结论

底盘是非全向、非阿克曼的左右轮差速底盘。手柄只应产生：

- `linear.x`：前进/后退；
- `angular.z`：左转/右转及原地旋转；
- `linear.y`：始终为 0，底盘不能横移。

实时参数证据：

- `holonomic_robot: false`；
- `max_vel_y: 0.0`、`min_vel_y: 0.0`、`vy_samples: 0`；
- 底盘发布 `/RosAria/lr_speed`，类型为左右轮速 `LRSpeed`；
- `/odom` 的有效速度通道是 `linear.x` 和 `angular.z`；
- 驱动节点为 `/chassis_server_node`，设备为 `/dev/udbot`；
- 导航 footprint 约为长 0.692 m、宽 0.52 m。

具体商业型号没有通过 ROS 参数或公开资料暴露。`/dev/udbot` 是设备别名，
不足以证明厂牌；但这不影响手柄运动轴配置。

## 当前硬件与 ROS 状态

底盘运行在独立计算机 `192.168.168.100`，ROS Master 为
`http://192.168.168.100:11311`。机械臂运行在 OrangePi
`192.168.168.120`，使用另一个 ROS Master。

底盘当前状态：

- `/chassis_control_state = 0`；
- `/chassis_work_mode = 0`；
- `/emergency = false`；
- `/plt_cmd_vel` 和 `/odom` 均为零速；
- 速度指令超时参数 `data_timeout = 0.2 s`；
- 速度平滑器运行频率 `100 Hz`；
- 平滑器硬参数上限为 `1.5 m/s`、`5.4 rad/s`；
- 当前导航上限为 `0.6 m/s`、`1.5 rad/s`。

## 底盘数据流

现有自动导航数据流：

`move_base/其他自动节点 -> /cmd_vel (Twist) -> velocity_smoother ->`
`/plt_cmd_vel (TwistStamped) -> chassis_server_node -> /dev/udbot -> 电机`

为手柄预留的数据流：

`手柄节点 -> /robot_cmd_vel (Twist) -> velocity_smoother ->`
`/plt_cmd_vel -> chassis_server_node`

`/robot_cmd_vel` 当前没有发布者，但 `velocity_smoother` 已订阅，因此它比
`/cmd_vel` 更适合作为手动输入。`/cmd_vel` 当前已有五个发布者，直接把手柄
接入该话题会产生控制竞争。首次带轮运动前仍应做零速度控制权测试，确认
`/robot_cmd_vel` 在厂商平滑器内的优先级与超时行为。

反馈通道包括：

- `/odom`：位姿和实际速度；
- `/RosAria/lr_speed`：左右轮速；
- `/battery`：电池；
- `/emergency`：底盘急停；
- `/chassis_sensors_info`：底盘传感器状态；
- `/chassis_control_state`、`/chassis_work_mode`：控制与工作模式；
- `/chassis_reply`：底盘命令回复。

## 机械臂数据流

OrangePi 当前运行 `joint_trajectory_controller`。已有 QP Servo 接口接收的
是目标 TF 帧动作，不是连续摇杆速度。MoveIt Servo 软件已安装，但节点尚未
启动。整机手柄控制应增加 MoveIt Servo 配置：

`手柄节点 -> TwistStamped/JointJog -> MoveIt Servo ->`
`joint_trajectory_controller -> Piper hardware -> CAN`

这样可保留关节限位、自碰撞和规划场景检查。不要把摇杆直接映射为电机电流
或无校验的关节位置。

## 已确定的第一版手柄映射

`Menu` 在底盘和机械臂两种模式间切换，两个运动输出始终互斥。

底盘模式：

- 左摇杆 X：转向 `angular.z`；
- RT：前进油门 `linear.x`；
- LT：刹车；
- `linear.y` 永远为 0；
- 暂不分配倒车操作，以免擅自改变用户给定的按键语义。

机械臂模式：

- 右摇杆左右/上下：末端左右、上下平移；
- RT/LT：末端向前/向后平移；
- 十字键左右/上下：末端 yaw/pitch 姿态微调；
- B：恢复原始末端姿态；
- A：切换夹爪开/关；
- Y：产生语音电梯模式请求，当前只预留接口，不启动语音或按钮流程。

这里的“右摇杆 4DOF”按四个摇杆方向实现；从机器人运动学看，右摇杆提供
两个平移自由度，两个扳机提供第三个平移自由度，十字键提供两个姿态自由度。
若需求实际是同时控制四个独立平移自由度，还需要再指定一个输入轴。

从机械臂模式切回底盘时，系统先停止机械臂并进入 `ARM_RETURNING`，调用回原位
接口。只有收到回原位成功回执后才启用底盘；失败时底盘继续锁定。

建议初始手动限速为 `0.25 m/s` 和 `0.6 rad/s`，输入频率 50 Hz，摇杆死区
0.10，使用斜坡限速。任何 100 ms 输入丢失都先发送零速；200 ms 是底盘自身
命令超时上限，不应把它当作主安全机制。

## 推荐部署结构

手柄可接到 OrangePi，也可接到同一局域网内的另一台设备。OrangePi 当前未检测
到手柄，且缺少 ROS `joy` 驱动，但已安装 `joy_teleop` 与 MoveIt Servo。正式接入
时增加三个独立进程：

1. 输入与安全监督进程：读取手柄、实施死区、使能键、超时和模式互锁；
2. 底盘发布进程：使用 `ROS_MASTER_URI=http://192.168.168.100:11311`，只发布
   `/robot_cmd_vel`；
3. 机械臂 Servo 进程：使用 OrangePi 本机 ROS Master，发布末端速度或关节
   Jog，并监听机械臂状态。

两个输出进程必须共享同一个手柄监督心跳，但分别连接两个 ROS Master。
自动按钮闭环、MoveIt 轨迹任务或底盘导航运行时，应由同一个控制权管理器拒绝
手动输出，不能依赖 ROS 多发布者的最后到达消息来仲裁。

当前已实现与硬件无关的映射和模式互锁，配置在 `gamepad_teleop.yaml`。总开关
`enabled` 和执行开关 `execute_enabled` 都是 `false`，所有底盘、Servo、回原位和
语音端口均标记为 `connected: false`，因此现阶段只能做 JSON 干运行。
