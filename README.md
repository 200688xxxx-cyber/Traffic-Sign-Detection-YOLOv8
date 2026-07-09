# 基于 YOLOv8s-SmallHead 的交通标志检测系统

## 项目简介

本项目基于 **YOLOv8s-SmallHead** 模型实现交通标志检测系统，采用 **TT100K** 数据集进行训练，并针对交通标志小目标检测场景进行了模型改进。

为了提升远距离、小尺寸交通标志的检测能力，本项目在原始 YOLOv8s 网络基础上新增 **P2 检测头（SmallHead）**，结合 Mosaic 数据增强、AdamW 优化器以及多种数据增强策略，对模型进行了优化。

最终模型在验证集上取得了：

| 指标 | 数值 |
|------|------|
| Precision | 74.7% |
| Recall | 72.4% |
| mAP@0.5 | 78.0% |
| mAP@0.5:0.95 | 52.0% |

同时，本项目完成了 **ONNX 模型导出**，在 CPU 环境下推理速度相比 PyTorch 原模型提升约 **72.11%**。

---

# 项目特点

- 基于 YOLOv8s-SmallHead 目标检测模型
- 新增 P2 检测头，提高小目标检测能力
- 使用 TT100K 交通标志数据集
- 支持 21 类交通标志检测
- 支持 ONNX 模型部署
- 提供 Web 在线检测系统
- 支持图片检测

---

# 项目目录

```
Traffic-Sign-Detection-YOLOv8
│
├── README.md
├── requirements.txt
│
├── train.py
├── predict.py
├── export.py
├── app.py
│
├── models
│   └── yolov8_smallhead.yaml
│
├── datasets
│   └── dataset.yaml
│
├── weights
│   └── best.pt
│
├── results
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxPR_curve.png
│   ├── BoxF1_curve.png
│   ├── BoxP_curve.png
│   └── BoxR_curve.png
│
├── images
│
└── report
    └── 实验报告.pdf
```

---

# 数据集

本项目采用 **TT100K（Tsinghua-Tencent 100K）交通标志数据集**。

数据集包含多种道路交通标志，覆盖：

- 限速标志
- 禁止标志
- 指示标志
- 警告标志
- 人行横道标志

本实验共选取 **21 个类别**进行训练。

---

# 开发环境

| 项目 | 配置 |
|------|------|
| OS | Windows 10/11 |
| Python | 3.10 |
| PyTorch | 2.x |
| CUDA | 可选 |
| Ultralytics | YOLOv8 |
| OpenCV | 4.x |

---

# 安装环境

安装依赖：

```bash
pip install -r requirements.txt
```

或者：

```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install onnxruntime
pip install streamlit
```

---

# 数据集准备

数据集采用 YOLO 格式：

```
datasets
│
├── images
│   ├── train
│   ├── val
│   └── test
│
└── labels
    ├── train
    ├── val
    └── test
```

修改 `dataset.yaml`：

```yaml
path: datasets

train: images/train
val: images/val
test: images/test

nc: 21

names:
  - i10
  - i12
  - i13
  - i2
  - i4
  - i5
  - il100
  - il60
  - il80
  - io
  - ip
  - p10
  - p11
  - p19
  - pg
  - ph4
  - ph4.5
  - pl100
  - pn
  - pne
  - w55
```

---

# 模型训练

运行：

```bash
python train.py
```

或者：

```bash
yolo detect train \
model=models/yolov8_smallhead.yaml \
data=datasets/dataset.yaml \
epochs=300 \
imgsz=640 \
batch=8
```

训练结束后模型保存在：

```
runs/detect/train/
```

---

# 模型预测

```bash
python predict.py
```

或者：

```bash
yolo detect predict \
model=weights/best.pt \
source=images/
```

预测结果保存在：

```
runs/detect/predict/
```

---

# ONNX 导出

导出：

```bash
python export.py
```

或者：

```bash
yolo export \
model=weights/best.pt \
format=onnx
```

导出成功后：

```
best.onnx
```

即可用于部署。

---

# Web 检测系统

启动：

```bash
streamlit run app.py
```

浏览器打开：

```
http://localhost:8501
```

上传图片即可完成交通标志检测。

---

# 实验结果

最终实验结果如下：

| 指标 | 数值 |
|------|------|
| Precision | 74.7% |
| Recall | 72.4% |
| mAP@0.5 | 78.0% |
| mAP@0.5:0.95 | 52.0% |

训练曲线：

```
results/results.png
```

混淆矩阵：

```
results/confusion_matrix.png
```

PR 曲线：

```
results/BoxPR_curve.png
```

F1 曲线：

```
results/BoxF1_curve.png
```

---

# 模型优化

本项目主要进行了以下改进：

- 新增 P2 检测头（SmallHead）
- Mosaic 数据增强
- HSV 色彩增强
- 高斯噪声增强
- 运动模糊增强
- AdamW 优化器
- 调整 Mosaic Scale
- ONNX 部署优化

---

# 部署效果

| 模型 | FPS |
|------|------|
| PyTorch | 3.95 |
| ONNX Runtime | 6.81 |

ONNX 推理速度相比 PyTorch 提高约 **72.11%**。

---

# 项目成员

课程：人工智能基础

项目：基于 YOLOv8s-SmallHead 的交通标志检测系统

成员：

- 成员1 杨诗钰
- 成员2 张瑜暄
- 成员3 高雅婷
- 成员4 周子勋
- 成员5 薛鈺玺

