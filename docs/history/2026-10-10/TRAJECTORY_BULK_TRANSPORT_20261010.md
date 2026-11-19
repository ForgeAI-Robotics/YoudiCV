# 完整轨迹传输修复

20261010_163103回位失败于preparing_trajectory：JSON压缩后85045字节仍大于58000字节预算，尚未发送运动指令。不是关节限位或起点碰撞。

现有TimedTrajectoryUdpClient API、GUI、token、目标IP和端口5010保持兼容。小轨迹使用原UDP编码；超过60000字节的轨迹使用TCP/5010，保留完整采样和原始浮点值。TCP与UDP可以使用同一个数字端口，不是同一个socket。客户端先检查UDP health返回trajectory_tcp_v1，旧桥缺能力则拒绝发送大轨迹，不自动回退为危险截断包。

TCP帧为4字节网络字节序长度加完整JSON，上限2MiB，header/body均循环收齐，有总读取超时；连接/请求数量有界，只允许原allowed_client IP和token。传输不完整、长度过大、解析失败不会调用轨迹处理器。该2MiB是资源上限，不再是单个UDP数据报大小；当前2..500点合法轨迹远低于此值。

OrangePi在接收全部数据后，执行原限位、点数、时长、速度和MoveIt整段检查，再走原控制模式、硬件反馈与动作看门狗。UDP/TCP共用同一busy锁，不可并行动作。时长仍≤15s、点数2..500，新增拒绝非有限时间戳，没有扩大关节限位、速度或碰撞范围。

TCP execute结果在同一桥进程缓存最近256个task_id和内容摘要；同ID同内容返回已有结果不重复运动，不同内容拒绝。validate每次重新检查场景。此缓存不是重启持久事务，不等于跨重启永远恰好一次。客户端不自动重发execute；如果发送后丢失回复，报告执行状态未知，应查状态/编码器再决定下一步，不能凭超时再次执行。

回原位规划不再为数据包大小提高速度；只在原时长/点数/速度约束需要时选既有速度档。抬升绕行方案也使用完整传输预算，按既有档位满足15s限制并检查整条路径。所有点的positions/velocities/accelerations/time仍传递给同一个轨迹控制器，不切割成若干运动命令，因此不会人为增加分段停顿。

## 验证

105项相关测试通过：TCP拆包、截断、超长header、缺能力、认证、busy共享、重复execute去重、重复validate保持新鲜、时间/关节限位、完整路径先检后执行，以及原回位/脱困/轨迹相关测试。

OrangePi实际5010只读验证：旧失败姿态生成279点，时长13.873273s，约129745字节，通过TCP全部接收，MoveIt全部279点校验通过，约0.731s；前后六关节完全相同。原UDP两点725字节的validate也通过，wire_transport=udp。没有发送任何execute请求，未验证这条长路径的实机跟踪表现。

证据目录：artifacts/diagnostics/20261010_bulk_trajectory_tcp，failed_pose_validation.json、udp_compat_validation.json和deployment_service.txt。

## 部署与回退

AGX：piper_handeye/timed_trajectory.py、trajectory_transport.py、run_direct_return_to_origin.py。
OrangePi：tools/orangepi_timed_trajectory_bridge.py、新增tools/piper_trajectory_transport.py，仅重启piper-timed-trajectory-bridge.service。既有开机服务同时监听TCP/UDP，不新增自动动作。
备份：version_snapshots/20261010_163502_before_bulk_trajectory_tcp，remote内为原OrangePi桥。无任务时恢复旧文件并重启该桥；保留新增helper也不会被旧桥调用。撤回后长轨迹单UDP大小问题会恢复。

TCP帧设计依据Python官方Socket HOWTO：流传输的单次recv不保证收齐一条消息，应使用明确长度和完整接收循环：https://docs.python.org/3/howto/sockets.html 。
