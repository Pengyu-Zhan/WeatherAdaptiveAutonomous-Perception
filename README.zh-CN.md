# WeatherAdaptiveAutonomous-Perception
[🇬🇧 English](README.md)
> 基于 YOLOv8 的天气自适应目标检测框架，采用门控特征融合机制，面向自动驾驶场景。

<!-- TODO: 训练完成后替换为实际效果图 -->
<!-- ![demo](assets/demo_result.png) -->

## 简介

WeatherAdaptiveAutonomous-Perception 是一个基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 构建的天气感知目标检测框架。核心贡献为 **EnvAdaptiveFusion** 模块——一种轻量级门控机制，能够根据实时天气条件（晴天、雨天、雾天、雪天、阴天）动态调整多尺度特征融合策略。

系统在 FPN Neck 的跳跃连接处（P3/P4/P5）嵌入 **WeatherGatedConcat** 机制，由天气分类分支生成逐层级的门控值，在特征拼接前对 Backbone 特征与 Neck 特征的融合比例进行自适应调制。

### 核心特性

- **EnvAdaptiveFusion 模块** — 在 FPN 跳跃连接处引入天气条件门控，生成可解释的逐层级门控值（G_P3 / G_P4 / G_P5）
- **WeatherGatedConcat** — 可学习的软门控层，在特征拼接前施加，使模型能在不同天气下自适应地强调不同尺度特征
- **天气分类分支** — 轻量级辅助分类器，输出天气类别 logits 用于驱动门控值生成
- **视频推理管线** — 自适应抽帧（Δn = ⌊f₀/fₛ⌋），支持天气 HUD 叠加显示与实时门控值可视化
- **ONNX 导出** — 支持模型导出，输入尺寸 `[1, 3, 640, 640]`，输出尺寸 `(1, 14, 8400)`

## 模型架构

```
                        输入图像
                          │
                    ┌─────┴─────┐
                    │  Backbone  │
                    │  (YOLOv8)  │
                    └─────┬─────┘
                          │
                 ┌────────┼────────┐
                 P3       P4       P5  (Backbone 特征图)
                 │        │        │
           ┌─────┴──┐ ┌──┴───┐ ┌──┴───┐
           │Weather  │ │Weather│ │Weather│
           │Gated    │ │Gated  │ │Gated  │
           │Concat   │ │Concat │ │Concat │
           └─────┬──┘ └──┬───┘ └──┬───┘
                 │        │        │
                 ▼        ▼        ▼
           ┌─────────────────────────────┐
           │    FPN Neck（改进后）         │
           └──────────────┬──────────────┘
                          │
                ┌─────────┼─────────┐
                │         │         │
              小目标     中目标     大目标
              检测头     检测头     检测头
                          │
                       检测输出

        ┌────────────────────────────────────┐
        │       天气分类分支                   │
        │  输入 → 天气 Logits → 门控值         │
        │  G_P3, G_P4, G_P5 ∈ [0, 1]        │
        └────────────────────────────────────┘
```

## 数据集

本项目使用 [BDD100K](https://www.bdd100k.com/) 数据集，经天气均衡采样后约 15k 张图像。

| 天气类别   | 数据划分  | 图像数量   |
|-----------|----------|-----------|
| 晴天      | train    | —         |
| 雨天      | train    | —         |
| 雾天      | train    | —         |
| 雪天      | train    | —         |
| 阴天      | train    | —         |

<!-- TODO: 补充实际数据分布 -->

**注意：** 本仓库不包含数据集。请从 [BDD100K 官网](https://www.bdd100k.com/) 下载数据，并按照 [`docs/data_preparation.md`](docs/data_preparation.md) 中的说明进行数据准备。

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7

### 安装步骤

```bash
git clone https://github.com/Pengyu-Zhan/WeatherAdaptive-Perception.git
cd WeatherAdaptive-Perception
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
python scripts/train.py --cfg configs/model/yolov8-envadaptive.yaml \
                        --data configs/data/bdd100k_weather.yaml \
                        --epochs 100 \
                        --batch-size 16
```

### 验证

```bash
python scripts/val.py --weights runs/train/best.pt \
                      --data configs/data/bdd100k_weather.yaml
```

### 视频推理

```bash
python scripts/video_inference.py --weights runs/train/best.pt \
                                  --source path/to/video.mp4 \
                                  --show-gates        # 开启门控值 HUD 叠加显示
```

### ONNX 导出

```bash
python scripts/export_onnx.py --weights runs/train/best.pt \
                              --imgsz 640
# 输出: 输入 [1, 3, 640, 640] → 输出 (1, 14, 8400)
```

### 门控值可视化

```bash
python scripts/visualize_gates.py --weights runs/train/best.pt \
                                  --source path/to/images/ \
                                  --save-dir results/gate_vis/
```

## 项目结构

```
WeatherAdaptive-Perception/
├── README.md
├── README_CN.md                # 中文文档
├── LICENSE
├── .gitignore
├── requirements.txt
├── configs/
│   ├── model/                  # 模型结构配置文件 (.yaml)
│   └── data/                   # 数据集配置文件 (.yaml)
├── models/
│   ├── env_adaptive_fusion.py  # EnvAdaptiveFusion 模块
│   ├── weather_gated_concat.py # WeatherGatedConcat 层
│   └── weather_classifier.py   # 天气分类分支
├── scripts/
│   ├── train.py                # 训练入口
│   ├── val.py                  # 验证 / 评估
│   ├── video_inference.py      # 视频推理（含天气 HUD）
│   ├── export_onnx.py          # ONNX 模型导出
│   └── visualize_gates.py      # 门控值可视化
├── tools/
│   ├── dataset_prepare.py      # BDD100K 预处理与天气采样
│   └── frame_extract.py        # 自适应抽帧工具
├── assets/                     # README 展示用图片
├── docs/                       # 补充文档
│   └── data_preparation.md     # 数据集准备指南
└── results/                    # 实验输出（已 gitignore）
```

## 实验结果

<!-- TODO: 训练完成后补充 -->

### 检测性能（BDD100K 验证集）

| 模型           | 天气条件 | mAP@50 | mAP@50:95 | 参数量 (M) | FLOPs (G) |
|---------------|---------|--------|-----------|-----------|-----------|
| YOLOv8n       | 全部     | —      | —         | —         | —         |
| YOLOv8n + Ours| 全部     | —      | —         | —         | —         |

### 门控响应分析

<!-- TODO: 补充 gate 可视化图 -->
<!-- ![gate_response](assets/gate_analysis.png) -->

## 部署

### Jetson TX2 NX（计划中）

- TensorRT FP16 优化
- 实时推理性能基准测试
- 边缘部署流水线

<!-- TODO: 部署完成后补充具体数据 -->

## 引用

如果本项目对您的研究有所帮助，欢迎引用：

```bibtex
@misc{zhan2025weatheradaptive,
  title={Weather-Adaptive Object Detection with Gated Feature Fusion},
  author={Zhan, Pengyu},
  year={2025},
  url={https://github.com/Pengyu-Zhan/WeatherAdaptive-Perception}
}
```

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BDD100K 数据集](https://www.bdd100k.com/)

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可协议。

`ultralytics_changes/` 目录下的代码基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 修改，遵循其原始 [AGPL-3.0 协议](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。所有修改内容均在源文件中明确标注。
