# YOLOv8 PCB 缺陷检测

本项目旨在使用 YOLOv8 目标检测模型检测印刷电路板（PCB）上的缺陷。按照以下步骤准备数据集、训练模型并进行推理和评估。

## 克隆项目

使用以下命令克隆项目到本地：
```bash
git clone https://github.com/lireno/yolov8-pcb-detection
```

## 项目结构

```
yolov8-pcb-detection/ 
├── data/ 
│   └── PCB/ 
│       ├── images/ 
│       │   ├── train/ 
│       │   └── val/ 
│       └── labels/ 
│           ├── train/ 
│           └── val/ 
├── runs/ 
│   ├── detect/ 
│   │   ├── predictions/ 
│   │   └── val/ 
│   ├── train/ 
│   │   └── weights/ 
│   ├── train2/ 
│   │   └── weights/ 
│   └── train3/ 
│       └── weights/ 
├── scripts/ 
│   └── convert_voc_to_yolo.py 
├── VOC2007/ [将你下载的 VOC 数据集放在这里]
│   ├── Annotations/ 
│   └── JPEGImages/ 
├── .gitignore 
├── evaluate.py 
├── infer_single.py 
├── predict.py 
├── README.md 
├── requirements.txt 
├── train.py 
├── yolo11n.pt 
└── yolov8n.pt
```

## 前置条件

1. 安装所需的 Python 包：
   ```bash
   pip install -r requirements.txt
   ```

2. 下载 VOC 数据集并将其放置在 `VOC2007` 文件夹中。

3. 更新 `PCB.yaml` 文件，将数据集路径设置为绝对路径：
   ```yaml
   path: D:\yolov8-pcb-detection\data\PCB  # 数据集的绝对路径
   ```
   **注意：** 确保路径中不包含中文字符，否则可能会导致错误。

## 项目运行步骤

### 第 0 步：准备数据集

下载 VOC 数据集并将其放置在 `VOC2007` 文件夹中。确保文件夹结构符合预期格式。

### 第 1 步：将 VOC 转换为 YOLO 格式

运行 `convert_voc_to_yolo.py` 脚本，将 VOC 数据集的标注转换为 YOLO 格式：
```bash
python scripts/convert_voc_to_yolo.py
```

### 第 2 步：训练模型

使用 `train.py` 脚本训练 YOLOv8 模型：
```bash
python train.py
```
训练好的权重将保存在 `runs/train/weights` 目录中。

### 第 3 步：评估模型

使用 `evaluate.py` 脚本评估训练好的模型：
```bash
python evaluate.py
```
该脚本将生成混淆矩阵并显示关键指标，例如精确率、召回率和 mAP。

### 第 4 步：对新图像进行预测

使用 `predict.py` 脚本对验证集或新图像进行预测：
```bash
python predict.py
```
预测结果将保存在 `runs/detect/predictions` 目录中。

### 第 5 步：对单张图像进行推理

使用 `infer_single.py` 脚本对单张图像进行推理：
```bash
python infer_single.py --image <path_to_image> --model <path_to_model_weights> --conf <confidence_threshold>
```
例如：
```bash
python infer_single.py --image data/PCB/images/val/sample.jpg --model runs/train2/weights/best.pt --conf 0.25
```
标注后的图像将保存在与输入图像相同的目录中，文件名后缀为 `_result`。

## 注意事项

- 确保脚本中的所有路径已更新为您系统的目录结构。
- 使用绝对路径而非相对路径，以避免潜在错误。
