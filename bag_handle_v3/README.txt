纸袋提手部署包 v3：使用历史 v7c 权重（不是 v16，也不是旧交付包中的 v3）。

远端目录：/home/qianjue/contral_test/bag_handle_v3
仅保存为独立版本；不替换 v2、不切换 GUI/服务、不启动相机、不控制机器人、不安装依赖。

best.pt：YOLO26m-Pose，v7c 第6轮最佳权重。
infer_bag.py：沿用 v2 的静态图片推理接口。
check_env.py：只读依赖与 CUDA 检查。
model_info.json：版本、来源、默认阈值、权重哈希。
examples/：一张纸袋、一张非袋柜子；*_local_reference.json 是本地 CPU 试跑，不是 Jetson 验证。
test_reports/：历史同口径测试摘要，不是机械臂抓取成功率。
training_reference/：训练参数、采样源码与旧报告，仅用于追溯，不是自包含训练集。

默认检测置信度0.65、关键点置信度0.5、imgsz640。
class0 且点置信度达标、坐标有限并在原图及预测框内时，输出绿框和一个抓取点；否则红框、无点。
坐标是原始输入图像像素坐标，左上角为原点。不要仅凭 model_status 执行抓取。
grasp_ready_2d 仅是2D输出门控，不保证袋子类别、提手材料位置、深度、碰撞或真实可抓取性。
没有深度或机械臂控制代码；没有新增“前/后提手”运行时过滤。

手动检查（均不启动相机/机械臂）：
  cd /home/qianjue/contral_test/bag_handle_v3
  sha256sum -c SHA256SUMS
  /home/qianjue/anaconda3/envs/elevator-yolo26/bin/python check_env.py
  /home/qianjue/anaconda3/envs/elevator-yolo26/bin/python infer_bag.py --source examples/input.jpg --device 0
  /home/qianjue/anaconda3/envs/elevator-yolo26/bin/python infer_bag.py --source examples/cabinet.jpg --device 0

使用现有 Jetson torch/torchvision/CUDA 环境，不要直接安装桌面版 CUDA wheel。
requirements-reference.txt 仅供版本参考，不会自动安装或升级任何依赖。
旧目录 bag_handle_yolo26_v3_20261013 是早期交付包，不是本次 v7c 的 v3；该目录保持不变。
