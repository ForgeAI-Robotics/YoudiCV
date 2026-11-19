# 优小妹机器人控制项目

电梯外呼、内呼按钮的视觉定位与机械臂闭环操作，包含语音任务、Android指令对接、AGX导航调度和DualSense手柄控制。

源码与本机运行数据分开管理；仓库不包含设备认证令牌、实机标定、下载模型、历史日志或代码备份。本次目录整理不改变机械臂运动、视觉判断或导航策略。

## 模块索引

查阅 [传统算法与运行流程索引](docs/模块索引.md)，按感知、规划、运动控制、业务流程和操作入口查找现有源码。生产电梯按键统一使用原闭环入口。

## 目录

```text
contral_test/
├── piper_handeye/                # 视觉、任务、规划、安全和通信模块
├── tools/                       # 桥接、GUI、诊断及维护工具
├── tests/                       # 自动化测试及小型固定夹具
├── config/examples/             # 去掉认证令牌的配置示例
├── deployment/                  # 跨设备部署文件和服务模板
├── systemd/                     # 本机服务模板
├── desktop/                     # 桌面启动器副本
├── docs/
│   ├── INDEX.md                 # 文档导航
│   ├── guides/                  # 使用与系统说明
│   └── history/YYYY-MM-DD/      # 按日期保留的设计与修改记录
├── robot_model_zero_render/     # 机器人模型与几何工具
├── button_light_evaluation/     # 离线灯态评估工具
├── run_*.py                     # 现有生产入口，保持原路径
├── requirements-handeye.txt
└── environment-handeye.yml
```

`artifacts/`、`archives/`、`third_party/`、本机配置和标定仍可在机器人上使用，但由`.gitignore`排除。根目录旧文档与日志目录的兼容软链接继续保留，并在文件管理器中隐藏；Git不收录这些兼容入口。模型权重不进入普通源码提交。

## 从源码开始

1. 先阅读[项目文档索引](docs/INDEX.md)，按设备与任务查找对应的部署说明。
2. 根据目标机已有的 ROS/MoveIt、相机 SDK 和 Python 环境，对照 [Conda 环境记录](environment-handeye.yml)与 [Python 依赖清单](requirements-handeye.txt)准备依赖。
3. 使用 [config/examples/](config/examples/) 中的模板核对本机配置，并从设备侧准备模型、标定和认证配置。
4. 先执行下方的无运动验证。需要实机联调时，再参考对应设备的服务模板和部署说明。

交付与恢复入口见[备份位置与恢复方法](docs/ARCHIVE_LOCATIONS.txt)。异常状态字段见[异常接口说明](docs/interfaces/exception_status_20261030.txt)。

## 主要入口

| 功能 | 入口 |
|---|---|
| 唯一生产按键闭环 | `run_button_light_closed_loop.py` |
| 按键目标分发 | `run_button_batch.py` |
| 常驻按钮视觉服务 | `run_button_vision_server.py` |
| 优小妹控制台及循环测试 | `run_voice_assistant_gui.py` |
| Android指令界面 | `run_android_command_gui.py` |
| AGX导航任务界面 | `tools/navigation_task_gui.py` |
| 手柄控制界面 | `tools/dualsense_input_gui.py` |
| 只读原位检查 | `run_origin_position_check.py` |
| 安全归位 | `run_direct_return_to_origin.py` |

`run_online_replan_no_contact.py`提供指向和接近能力，不是独立完整按键入口。语音、手动任务、导航联动与循环测试应复用生产闭环。

## 环境和本机配置

依赖记录见`requirements-handeye.txt`和`environment-handeye.yml`。实际部署还需要对应设备的ROS/MoveIt、相机SDK、模型、桥接服务和标定；仅下载仓库不能直接控制硬件。

- 示例配置位于`config/examples/`，其中`SET_LOCAL_TOKEN`需要替换为设备实际认证配置。
- 本机`piper_button_pose.yaml`、`gamepad_teleop.yaml`、`current_calibration/`和导航配置保留原位置，不因整理而重置。
- 模型和SDK保留于`third_party/`及已有模型目录；不清理使用中的共享库或Conda环境。
- 其他部署和跨设备接口说明见[文档索引](docs/INDEX.md)。

## AGX 导航工具的 SSH 配置

AGX 导航界面和只读记录工具使用 SSH 公钥认证。将设备主机密钥指纹与现场记录核对后，加入当前用户的 `known_hosts`。通过环境变量指定地址和用户名：

```bash
export YOUDI_NAV_SSH_HOST=robot-agx.example
export YOUDI_NAV_SSH_USER=ubuntu
```

使用 SSH agent 或默认私钥。私钥保存在其他位置时，可设置 `YOUDI_NAV_SSH_KEY_FILE` 指向本机文件。部署账户需要有权执行对应的导航查询或暂停操作。凭据文件留在部署设备。

## 无运动验证

在已准备依赖的开发环境中运行，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_exception_reporting.py tests/test_arm_execution_report.py \
  tests/test_mission_status.py tests/test_button_light_flow.py
```

真实硬件脚本不作为普通测试运行。接口接受、规划成功和真实运动完成分别判断；归位以实际反馈核验为准，不能用程序退出代替。

固定回归素材保存在 `tests/fixtures/`，与可清空的运行日志分开。已验证的布局原图和冻结结果位于 `tests/fixtures/numeric_panel_layout_offline/`；清日志时不要删除这些测试输入。

## 日志、备份和初始状态

新任务日志位于`artifacts/runs/`，图像位于`artifacts/captures/`，试验与录像位于`artifacts/trials/`、`artifacts/recordings/`。历史清空后保留空目录，新任务按原入口生成记录。

2026-10-30归零前的日志及缓存备份：

```text
/home/qianjue/robot_project_backups/20261030_日志缓存归零前/
```

其中保存完整ZIP、逐文件SHA256清单及恢复工具。备份单独传输，不提交到Git。历史代码、修改前快照和交付成果已集中保存到桌面`20261030_项目交付与代码备份`，项目内的重复归档及39份旧工作副本已在校验后清除，必要的空目录保留。

桌面`代码及回退/历史工作副本与补充资料_20261030.zip`保存被移出的旧开发副本及补充资料，可用同目录`restore_history.py`恢复。运行/回归工具仍引用的RGB-D素材、评测数据及当前模型没有清理。详见[备份位置与恢复方法](docs/ARCHIVE_LOCATIONS.txt)。

`tools/reset_runtime_data.py`默认只预览。重置须显式指定项目外备份位置，校验通过后进行；不发送机器人指令、不清停止锁、不删除模型或标定。详见[目录与备份说明](docs/PROJECT_LAYOUT.txt)。

## 版本与发布

已配置Git忽略规则和LF换行规范。提交限于源码、文档、配置示例、服务模板及必要的小型测试夹具。

本次不上传GitHub、不创建远端仓库，也不擅自指定开源许可证。对外发布权限和第三方依赖许可证按双方约定确认。
