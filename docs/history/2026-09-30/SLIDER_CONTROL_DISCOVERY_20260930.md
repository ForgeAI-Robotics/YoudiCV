# 滑块驱动接口追查补充（2026-09-30）

本次只读检查：未调用滑块相关服务、未发布速度/运动命令、未操作机械臂或底盘。

## 目前新增的证据

### 底盘 ROS 图

- 当前 `192.168.168.100:11311` 的 ROS Master 可访问。`/auto_container_manager/auto_container_cmd_srv` 正由 `/auto_container_manager_node` 提供，类型是 `custom_msgs_srvs/String`；`/auto_container_manager/auto_container_status` 类型为 `std_msgs/String`，由该节点发布并由 `/udp_commit_node` 订阅。
- `/auto_container_manager_node` 同时发布 `/cmd_vel`、`/laser_scan`、`/marker_pose`、点云特征和路径信息，并订阅 `/depth_camera_falling_point_base_cloud`。参数 `/UD_use_auto_container_manager` 与节点私有参数 `/auto_container_manager_node` 均存在；前者当前为 `true`。这些接口和运行参数支持“货柜/标记感知及底盘靠近”的解释，但尚不能证明服务绝不包含滑盖命令。
- 本机没有 `custom_msgs_srvs` 的 `.srv` 源定义，因此无法在此确认 `String` 服务请求/响应字段语义。没有调用服务读取响应，以免意外触发动作。

### Android 安装包与厂商 Binder 通道

检查的安装包：`Desktop/2026_0921_1044_SWJ.ARM.01V001.002.04@ca9f7486e14e262e5b03834cf4bf8cf3.apk.1`。

- APK 通过 Android Binder 绑定外部 `com.example.irducserver.RducService`（Intent action `com.connect.rducServer`），接口提供 `sendData(destination, priority, message)`，所以 Android 到设备控制器还存在 ROS/UDP 之外的厂商 IPC 通道。
- APK 的 AGX `active_result` 回调中，`command=3` 且 `value="1"` 或 `"0"` 时，会分别通过该 Binder 向目的节点 `255`、优先级 `5` 发送十六进制文本帧 `1303435502` 或 `1303434C02`。这是目前找到的最具体候选控制路径；帧含义和目的节点映射尚未从 APK 得知，不能直接称为滑块开/关命令。
- `MainActivity$2.receiveDataBySrc()` 当前为空实现；APK 本身没有公开这些帧的说明。Android 上的 `RducService` 实现不在该 APK 内，必须读取该服务 APK/厂商协议或服务日志才能继续解码。

### USB 串口与“自动货柜”封装

- APK 的 `/dev/ttyUSBCH3410` 串口速率为 115200。其 `sendCommand(mode,color)` 构造的帧明确记录为灯光模式/颜色；业务调用点只在启动和楼层选择时设置灯光。因此这条串口路径目前不支持“滑块电机由该串口控制”的判断。
- APK 中确有 `AutoContainerCmd` 类、对 `/auto_container_manager/auto_container_cmd_srv` 的通用请求封装，以及状态包分支；但本次静态检查没有找到 RobotArm 业务类实例化或调用该封装，也没有找到业务层注册 auto-container 状态监听器的调用点。因此它可能是遗留/未接线代码，不能用它单独证明 Android 应用实际在控制滑盖。

## 修正后的结论

现在有三条应分别追踪的控制域：底盘 ROS 的货柜感知/靠近服务；Android 的 RDUC Binder 私有帧通道；APK 上独立的 USB 串口指示灯通道。当前没有证据确认滑块电机挂在哪条链路上。优先级最高的新线索是 Binder `command=3` 的两条帧，因为它们是应用业务层确实调用的离散控制输出；但在获得 RDUC 服务端协议/日志或控制板接线前，不能把它们映射为滑盖动作，也不能发送试探帧。

## 下一步安全取证

1. 从 Android 设备导出已安装的 `com.example.irducserver` 服务 APK（或厂商给出对应源码/协议），检查 `sendData` 最终写往哪个总线/设备，以及帧 `1303435502`、`1303434C02` 的定义。
2. 从导航/AGX 端的只读日志核对 `active_result` 的 command 3 真实触发来源和 `value` 含义；当前本地日志没有这类事件样本。
3. 取得滑块电机驱动板的型号、线束照片/接线图，再把驱动板与 Android 服务、底盘 Jetson UART/GPIO/CAN/USB 逐项对应。
4. 继续保持只读；在用户明确要求测试前，不发送 command 3、不调用货柜服务、不驱动滑块。
