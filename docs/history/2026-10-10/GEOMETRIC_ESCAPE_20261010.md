# 2026-10-10 完整网格脱困修复

旧判据使用 GetStateValidity 返回的最大 contact.depth。单碰撞对通常只返回一个接触，接触点/三角形变化时深度会跳变，不能据此判断整段运动在加深还是脱离机壳。

生产入口改为 top_chassis_recovery.select_geometric_j2_j3_escape_target。只读 piper_escape_geometry.cpp 使用当前 robot_description 的完整 link2/顶板网格、固定变换及场景 padding，在模型副本上求向上平移至不相交所需距离；脱离后使用 FCL 两网格最小距离。前者是沿指定方向的退出距离，不是全局最小穿透深度，也不是实物接触传感器。模型不准确仍需重新标定，不能把模型通过等同于实物已安全。

J2增加、J3等量减少，其他关节不动。每0.001rad整段预检，退出距离不得增加，分离间隙不得降低（10µm计算容差）；MoveIt保持其他碰撞限制，脱离后禁止重新进入碰撞。FK须确认肘部与末端全程向上且总体至少抬升1mm。上限从0.050rad变为0.080rad：当前姿态0.050rad只有3.927mm间隙，不满足5mm；0.064rad达到7.151mm。规划预留7mm，实际终点必须≥5mm，未达不允许继续普通回位。速度仍为0.020rad/s。

运行时反馈、固定关节漂移、跟踪误差、几何趋势持续检查。趋势相对历史最佳值最多允许0.5mm编码器/数值波动，不能逐步积累后退。预检期间编码器变动>0.003rad拒绝执行。模型描述发生变化立即拒绝，重启只读worker后才能使用新模型。worker按需启动并阻塞等待请求，不增加空闲推理循环；源码用 tools/build_piper_escape_geometry.sh 在已装有Noetic MoveIt开发包的OrangePi上编译。控制桥重启时清理worker。

81项相关测试通过，新增覆盖接触深度跳变、真实几何恶化、其他碰撞对、净空不足、碰撞重新进入、度量缺失、常驻worker协议及异常退出。当前真实姿态完整模型预检约5.53秒（含冷启动），64步目标；历史姿态约2.89秒。客户端超时改为20秒，覆盖预检和低速运动，超时不重发动作。

从预测脱困终点到原位，5010 dry_run验证34点完整路径通过，时长1.62秒；未发送execute或recover动作。证据：artifacts/diagnostics/20261010_geometric_escape/{validation_output.txt,return_validation.json,mesh_clearance_curve.txt}。

本次没有实机运动。下一次返原位才会执行新的脱困；实际结果需要核对编码器、净空、桥日志和回原位报告。

## 回退

备份：version_snapshots/20261010_154203_before_geometric_escape，remote子目录保存部署前OrangePi副本，arm_gamepad_udp.py为AGX客户端原文件。OrangePi也保存于 ~/mobile_arm_demo/backups/20261010_154203_before_geometric_escape。

无任务/手柄控制时恢复对应文件并重启 piper-gamepad-servo-bridge 用户服务（具体名称见部署日志），AGX恢复 tools/top_chassis_recovery.py、tools/orangepi_gamepad_servo_bridge.py 和 piper_handeye/arm_gamepad_udp.py。新几何worker文件可保留但旧桥不调用。J3+2度和此前完整回程预检不在本次回退范围，不得误撤销。回退后旧深度跳变导致拒绝的问题会重新出现。

## 15:59 后续执行模式修复

155649实机尝试：旧报告为tracking diverged，编码器未移动。只读CAN连续8帧`2A1#0200010200000000`表明当前模式02。发现恢复动作先于CAN执行模式切换，导致仅ROS控制器切换后就发布抬升目标；电机Active并不代表CAN执行模式01。新增confirm_escape_actuation_mode：必须已全路径预检、ROS当前位姿保持，确认position running/trajectory stopped后发送既有模式帧，至少三帧新反馈01，漂移≤.003rad，之后才发布抬升目标。不能绕过碰撞预检去调用普通模式启用函数。失败审计包含恢复阶段和实际/目标关节。89项相关测试通过；部署仅重启5015，无实机运动。该证据支持模式顺序问题，但旧失败时未采集同步CAN，最终因果验证仍需下一次动作的模式/编码器日志。

备份version_snapshots/20261010_155914_before_escape_can_mode含remote原文件；本补丁仅orangepi_gamepad_servo_bridge.py与orangepi_arm_mode.py，不撤销之前几何修复。

## 实机验证 16:01

第一次20261010_160144失败为模式切换期间位姿漂移0.004992rad，超过0.003rad保持阈值，未继续脱困。故障采样J3约0.001082rad，相比此前+0.031451rad已改变，不能仅用放宽阈值解决；切换中的精确轨迹仍缺同步证据，不能断言只是噪声或已确认底层限位。
第二次20261010_160158成功：切换前后三帧CAN均01，保持漂移0；J2/J3各规划0.064rad，实际模型净空6.829mm；正常回位最终最大编码器误差0.00122173rad（0.070°），hardware_ok=1。证明完整脱困+回位至少成功一次，不证明首次模式转换已可靠。未为规避首次漂移扩大阈值或取消碰撞。

## 稳定后自动重新预检（第二次成功经验）

16:01第二次成功说明模式稳定后的实际位置可用于完整脱困，但首次切换导致J3变化的底层原因尚未完全确认。不能声称该变化是噪声或正常。保留0.003rad保持漂移阈值，旧计划因此中止，新增类型EscapeModeDriftError仅表示模式命令已经发送且新反馈三帧01，位置起点已失效。

仅该类型可以等待并重新预检一次。最多2秒内，反馈连续至少0.30秒相对稳定窗口起点变化≤0.001rad；模式始终01，原碰撞对以外碰撞拒绝，完整网格净空指标相对最佳值恶化>0.5mm拒绝。被动变化限定J2不下移/J3不外伸（1mrad反馈容差）、各≤0.080rad，固定关节变化≤0.010rad且不越原关节限位。该被动变化检查不授权发送运动目标。稳定后从新编码器位置重新运行完整路径、FK及几何预检，最多第二次尝试，不复用旧计划；新的模式保持/预检/执行若失败则结束。

模式已01时读取两批反馈确认，不重复发送模式命令。成功审计保留首次失败、稳定位置、观测首末值/次数；异常审计区分settling与重新预检。AGX失败报告保存桥诊断，不再只写reading_live_state。接收UDP缓冲扩至65535字节，恢复等待30秒覆盖两轮预检，不改变动作速度。

96项相关测试通过，模拟覆盖自动第二次、禁止第三次、持续运动超时、新碰撞、错误方向和模式确认失败。只重启5015桥；关节位置未变化。没有进行新实体脱困试验，下一次首次回位需核对activation_replan字段。

备份：version_snapshots/20261010_161213_before_escape_stable_replan（remote原桥、模式helper、top_chassis helper；本机客户端/回位脚本/AGENTS均有备份）。恢复对应文件并在无任务时重启5015桥即可撤回本次改动。
