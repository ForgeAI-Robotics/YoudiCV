纸袋提手识别交付 v2（模型训练修订v6，不是旧v3交付包）

存放目录：/home/qianjue/contral_test/bag_handle_v2
初次交付仅存放文件。2026-10-15 已接入项目纸袋检测 GUI：默认本目录 best.pt，conf=0.65、point_conf=0.5；infer_bag.py 新增复用常驻模型的 predict_frame，原 predict_image 保留。通过现有视觉服务只读 preview 获取画面，不控制机器人。GUI 入口及验证见项目根目录 BAG_DETECTION_GUI.md。

文件：best.pt最佳权重；infer_bag.py单张图片推理；check_env.py只读环境检查；model_info.json配置与校验信息；examples示例；test_reports测试；training_reference追溯。

默认检测置信度0.65，关键点置信度0.5。
绿框+点必须同时满足：class0、坐标有限、关键点置信度达标且点位在原图和框内；否则红框、无点、grasp_ready_2d=false。
model_status保留网络原始分类，status添加关键点检查；不能仅凭model_status授权抓取。
grasp_ready_2d只表示2D输出通过检查，不代表真实物体一定是袋子、点位准确、深度有效或允许运动。

手动验证（不控制机器人）：
  cd /home/qianjue/contral_test/bag_handle_v2
  /home/qianjue/anaconda3/envs/elevator-yolo26/bin/python check_env.py
  /home/qianjue/anaconda3/envs/elevator-yolo26/bin/python infer_bag.py --source examples/paper.jpg --device 0
系统python3可能缺少依赖；requirements-reference.txt仅供参考，不自动安装或升级Jetson的torch。

测试：固定conf=0.65、batch=8，145图（45袋子+100负样本）。100负样本0绿框/点，1允许红框；40可抓取目标检出38个，绿框精确率38/41=92.7%，点位容差通过33/41=80.5%。
点位容差为最长边归一化225像素后10像素，不是实际抓取成功率；标签未独立人工验收，近重复分组未全面审计。
示例*_local_batch_reference.jpg是本地批处理结果，单张/批处理和Jetson运行时可不同；Jetson推理验证情况以单独传输回执为准。

完整性检查：sha256sum -c SHA256SUMS（已更新 GUI 接入后的源码与本说明；权重不变）。
原始交付文件和校验清单：../version_snapshots/20261015_142800_before_bag_v2_gui/bag_handle_v2。
model_info.json 保留初次交付状态，接入及 Jetson 运行时验证以项目文档和 artifacts/diagnostics/bag_v2_upgrade/runtime_validation.json 为准。
