# YOLO + OpenCV 按钮亮灯批量评测

该工具复用当前生产代码：YOLO26m 定位指定按钮，随后用 OpenCV 比较同一按钮框内中心区域与外围区域的亮度。它不会连接相机或机械臂。

数据清单可用 CSV 或 JSONL，每行必须包含：

- `image`：图片路径；相对路径以清单所在目录为基准。
- `target`：目标按钮标签，例如 `10`、`B1`。
- `state`：真值 `ON`、`OFF` 或 `UNCERTAIN`。
- `gt_x1,gt_y1,gt_x2,gt_y2`：可选的按钮真值框；四项必须一起提供。

若一批图片属于同一个按钮，可先按 `dataset/ON`、`dataset/OFF`、`dataset/UNCERTAIN` 放置，再自动生成清单：

```bash
/home/qianjue/anaconda3/envs/elevator-yolo26/bin/python \
  button_light_evaluation/make_manifest.py /path/to/dataset \
  --target 10 --output /path/to/manifest.csv
```

运行：

```bash
cd /home/qianjue/contral_test
button_light_evaluation/run_eval.sh /path/to/manifest.csv --batch-size 16
```

指定固定输出目录后可断点续跑：

```bash
button_light_evaluation/run_eval.sh /path/to/manifest.csv \
  --output /path/to/eval-output --batch-size 16
```

输出包含 `predictions.jsonl`、`summary.json` 和 `mistakes.csv`。`summary.json` 分别报告目标检测率、可选的 IoU@0.5 召回率、包括拒识在内的严格状态准确率及其 Wilson 95% 置信区间、只统计 ON/OFF 决策的准确率与覆盖率、ON/OFF 分类别 precision/recall/F1、平衡准确率和吞吐量。报告同时保存模型 SHA-256 与全部阈值，保证结果可追溯。

先用 `--limit 100` 做小批检查，再去掉限制运行完整数据集。若要重建固定输出目录中的结果，使用新的输出目录或传入 `--no-resume`。
