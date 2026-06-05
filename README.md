# ThermoFusionNet

这是一个用于红外图像与可见光图像多模态融合的PyTorch项目，适合作为图形学课程大作业。

**注意：本项目不是分类任务**，只需要成对的可见光和红外图像，不需要分类标签、检测框或语义分割标签。

## 训练完成情况

- 训练环境：服务器（Autodl）
- 训练配置：`configs/train_4090.yaml`
  - 图像尺寸：256×256
  - 批次大小：8
  - 训练轮数：100 epoch
  - 学习率：0.0001
  - 损失函数权重：
    - 亮度损失：1.0
    - 梯度损失：5.0
    - 色彩损失：1.0
    - 热损失：2.0
- 最终损失：0.1-0.2
- 测试结果：已生成 `results/test_4090/`，包含融合图像和对比图

## 项目简介

ThermoFusionNet 采用双分支Encoder-Decoder架构，能够融合可见光图像的色彩和纹理信息，同时突出红外图像中的热源区域。

## 网络结构

- **Visible Encoder**: 处理RGB可见光图像，提取多尺度特征
- **Infrared Encoder**: 处理灰度红外图像，提取多尺度特征
- **Attention Fusion**: 简单稳定的注意力融合模块，生成注意力权重
- **U-Net Decoder**: 使用跳跃连接，融合特征并重建RGB图像

## 文件结构

```
ThermoFusionNet/
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── infer.py
├── demo_local.py
├── configs/
│   ├── train_4090.yaml
│   ├── train_local.yaml
│   └── train_overfit.yaml
├── models/
│   ├── __init__.py
│   ├── encoders.py
│   ├── attention_fusion.py
│   ├── decoder.py
│   └── thermofusion_net.py
├── losses/
│   ├── __init__.py
│   └── fusion_loss.py
├── datasets/
│   ├── __init__.py
│   └── ir_visible_dataset.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   ├── checkpoint.py
│   ├── logger.py
│   └── visualization.py
├── tools/
│   ├── __init__.py
│   ├── prepare_msrs.py
│   ├── create_mini_dataset.py
│   └── smoke_test.py
├── checkpoints/
├── results/
├── logs/
├── data/
│   ├── train/
│   │   ├── visible/
│   │   └── infrared/
│   ├── val/
│   │   ├── visible/
│   │   └── infrared/
│   └── test/
│       ├── visible/
│       └── infrared/
└── MSRS-main/  (需要自行准备)
    ├── train/
    │   ├── vi/
    │   └── ir/
    └── test/
        ├── vi/
        └── ir/
```

## 环境安装

```bash
pip install -r requirements.txt
```

## MSRS 数据集使用说明

本项目只使用 MSRS 数据集中的可见光和红外图像对：
- `MSRS-main/train/vi/` 和 `MSRS-main/train/ir/`
- `MSRS-main/test/vi/` 和 `MSRS-main/test/ir/`

暂不使用：
- detection 文件夹
- labels 或 Segmentation_labels

### 整理 MSRS 数据集

```bash
python tools/prepare_msrs.py --msrs_root ./MSRS-main --out_root ./data --val_ratio 0.1 --seed 42
```

参数说明：
- `--msrs_root`: MSRS 数据集根目录
- `--out_root`: 输出目录
- `--val_ratio`: 验证集比例（默认0.1）
- `--seed`: 随机种子（默认42）
- `--overwrite`: 覆盖现有文件（可选）

### 创建小数据集用于本地调试

```bash
python tools/create_mini_dataset.py --src_root ./data --out_root ./mini_data --num_train 20 --num_val 5 --num_test 5 --seed 42
```

### 创建过拟合测试数据集（用于快速验证）

```bash
python tools/create_mini_dataset.py --src_root ./data --out_root ./mini_overfit_data --num_train 3 --num_val 1 --num_test 1 --seed 42
```

## 本地调试流程

### 1. Smoke Test（快速检查代码链路）

```bash
python tools/smoke_test.py --config configs/train_local.yaml
```

Smoke Test 会：
- 读取配置
- 加载一个batch数据
- 前向传播
- 计算损失
- 反向传播
- 保存对比图

### 2. Overfit Test（过拟合测试，检查数据和训练流程）

```bash
python train.py --config configs/train_overfit.yaml
```

使用1-3对图像训练50 epoch，观察损失是否明显下降。如果无法过拟合，可能存在以下问题：
- 数据配对错误
- 归一化问题
- 损失函数问题
- 模型输出范围问题
- 梯度或学习率问题

### 3. Mini Training（小规模训练）

```bash
python train.py --config configs/train_local.yaml
```

使用20对左右mini_data训练，观察TensorBoard loss和results/local下的compare图。

### 4. 本地 Demo

```bash
python demo_local.py --config configs/train_local.yaml --checkpoint checkpoints/local/best.pth --output_dir results/local_demo
```

## TensorBoard

```bash
tensorboard --logdir logs
```

或针对特定配置的日志：

```bash
tensorboard --logdir logs/local
tensorboard --logdir logs/overfit
```

## 正式训练

```bash
python train.py --config configs/train_4090.yaml
```

断点续训：

```bash
python train.py --config configs/train_4090.yaml --resume checkpoints/4090/latest.pth
```

## 测试

```bash
python test.py --config configs/train_4090.yaml --checkpoint checkpoints/best.pth --output_dir results/test_4090
```

测试结果保存在指定output_dir下的fused和compare子目录。

## 单张推理

```bash
python infer.py \
  --visible path/to/visible.png \
  --infrared path/to/infrared.png \
  --checkpoint checkpoints/best.pth \
  --output results/infer_result.png \
  --img_size 256
```

会保存融合图像和对比图。

## 损失函数

- **Intensity Loss**: 让融合图像的亮度接近可见光和红外的最大值
- **Gradient Loss**: 使用Sobel算子保持边缘信息
- **Color Loss**: 保持可见光图像的色彩信息
- **Thermal Loss**: 在红外高亮区域（top 20%）突出热源

## 输出结果说明

- `checkpoints/`: 保存训练过程中的模型权重
- `results/train/`: 训练过程中的可视化结果
- `results/test/fused/`: 测试集的融合图像
- `results/test/comparison/`: 测试集的四列对比图（Visible | Infrared | Attention | Fused）
- `logs/`: TensorBoard日志文件
