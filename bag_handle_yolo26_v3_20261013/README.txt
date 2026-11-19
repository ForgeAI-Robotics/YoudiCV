纸袋提手 YOLO26m-Pose v3 交付包

远端目录：/home/qianjue/contral_test/bag_handle_yolo26_v3_20261013
此阶段只传文件，不安装依赖，不启动相机，不重启任何现有服务，不调用机器人。

主要文件
best.pt                  本次修正标注训练的最佳权重（100轮，325训练/55验证）
infer_bag.py             独立图片推理入口，JSON输出每个袋子的状态、框、提手中心像素坐标
check_env.py             只读检查Python、PyTorch、CUDA、Ultralytics及OpenCV环境
requirements.txt         训练时Ultralytics版本参考，不可直接覆盖Jetson已有Torch环境
model_info.json          类别、关键点定义、阈值、权重校验和、训练测试说明
examples/input.jpg       本地独立测试集第2张原图（多个袋子）
examples/reference.json  对应参考标注（助手标注，未由用户逐点确认）
examples/local_result.json / local_overlay.jpg   本地脚本试跑的输出，不代表Jetson已验证
training_reference/      本次训练及测试分析源码，仅供追溯；重训还需原始数据和预训练权重
test_reports/            测试指标及点位误差报告
SHA256SUMS               交付文件传输完整性校验

环境确认后可运行（以下命令不会控制机器人）：
  cd /home/qianjue/contral_test/bag_handle_yolo26_v3_20261013
  python3 check_env.py
  python3 infer_bag.py --source examples/input.jpg --device cpu
GPU版本环境确认后可把 --device cpu 改为 --device 0。
自有图片：
  python3 infer_bag.py --source /absolute/path/photo.jpg --device 0 --output-json /tmp/bag_result.json --save-overlay /tmp/bag_result.jpg
输出路径如已存在会拒绝覆盖。不要在正在运行按钮视觉服务的环境中直接升级依赖。

输出定义
handle_center_xy=[x,y]，为原始输入图像坐标，左上为(0,0)，x向右、y向下，单位像素。
每个袋子分别输出；invalid/uncertain无抓取点（null）。检测有效但关键点低置信度/越界时输出uncertain。
model_status是网络预测类别；status加了关键点可用性检查。此处不是完整“合规性判定器”。
只有2D坐标，不包含深度、相机内参、手眼标定、机械臂姿态或控制命令。
重合提手是允许的目标，但模型不能保证始终选择前侧提手，不能仅凭status=valid授权运动。

已知限制
训练候选中排除了uncertain样本，因此模型对uncertain类别的能力不可靠；不能靠该类别作为机器人安全依据。
50测试图片中45张做定量评估；参考点为助手人工标注，尚未独立人工确认。
原图多数为低分辨率商品图；外卖袋现场、相机画面及遮挡重合场景还需测试。
新模型整体框mAP50约46.9%，可抓取类关键点mAP50约92.2%，后者不是抓取成功率。
计入漏检，参考点误差<=5像素为67.5%，<=10像素为77.5%。
