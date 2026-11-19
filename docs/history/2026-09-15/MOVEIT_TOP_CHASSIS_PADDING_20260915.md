# MoveIt 安装机壳几何与边界恢复

日期：2026-09-15

OrangePi 运行时 `/robot_description` 已包含 `top_chassis_link`，其 visual 和
collision 都引用 `package://piper_description/meshes/top_chassis.STL`。SRDF 只放行
机壳与固定安装相邻的 `piper_base_link`、`piper_link1`；其他连杆与机壳默认参与
MoveIt 自碰撞检查。

2026-09-22 根据实机边界锁死结果修订：URDF 已按实际安装位置标定到原位约
5 mm 的几何间隙，因此不再额外重复膨胀 5 mm。验证器首次查询前向 MoveIt
PlanningScene 写入并读回：

```text
LinkPadding(link_name="top_chassis_link", padding=0.001)
```

该余量只处理建模和编码器小误差。当前机壳滑块/抽屉关闭：visual 继续保留
`top_chassis.STL`，collision 使用 `top_chassis_closed.STL`，删除厂商网格中只在
打开状态伸出的第二大独立连通部件；其余七个实体全部保留。`/apply_planning_scene`、
`/get_planning_scene` 不可用、写入失败或读回值不是 0.001 m 时，轨迹验证失败关闭。若 `/check_state_validity` 连接因
move_group 重启而失败，验证器清除本地已应用标志，下一次验证会重新写入并校验。

部署文件：

`/home/orangepi/mobile_arm_demo/tools/piper_moveit_collision.py`

部署前备份：

`/home/orangepi/mobile_arm_demo/tools/piper_moveit_collision.py.before_top_chassis_padding_20260915`

无运动验证使用按钮 8 已实测的记录原位、抬起观察位及反向返回，共查询 57 个
关节插值样本。2026-09-22 部署后的结果为 `moveit_collision_validated=true`、
`moveit_padded_link=top_chassis_link`、`moveit_link_padding_m=0.001`。

若断电手动移动或编码器边界误差使当前状态已经位于碰撞边界，普通控制仍然
fail-closed。唯一例外是 5015 桥的受限恢复命令：只有唯一接触对为
`piper_link2/top_chassis_link` 时，才允许 J2 向增加间隙方向以 0.020 rad/s、
最多 0.050 rad 移动；任何其他关节漂移、其他接触、硬件非 Active、反馈丢失或
末端仍无效都会停止。进入有效区后立即恢复标准轨迹控制器和完整 MoveIt 校验。

随后移除了 MoveIt 之外的 `piper_clearance.py` 高度代理：不再要求肘部/TCP 先
抬高 10 cm，也不再限制低位时第一关节只能摆动 6 度。回原位入口改为生成直接
轨迹并交由真实 MoveIt 网格逐段验证。运行时世界场景没有额外模拟箱体。

相机下方的 `piper_camera_base_link` collision 仍引用
`package://piper_description/meshes/piper_camera.STL`，没有删除或禁用。最终无运动
验证报告为 `clearance_return_runs/20260915_101605/report.yaml`，七个插值姿态均由
新 OrangePi 验证器通过。
