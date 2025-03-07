# 混合精度与知识蒸馏协同优化项目

## 项目简介
本项目旨在研究低资源环境下，通过混合精度计算与仅教师模型指导的知识蒸馏策略，对深度学习模型进行优化。项目支持在高性能服务器和边缘设备（通过 TensorRT 部署）上运行。实验数据集包括 CIFAR-10 和 ImageNet，确保实验结果具有广泛的代表性。

## 文件结构

project/
├── config.yaml              # 配置文件，保存训练超参数、数据集类型、路径等设置
├── main.py                  # 主程序入口，根据配置加载数据、模型、启动训练或推理
├── trainer/                 # 训练模块目录，包含训练逻辑、损失计算、日志记录等
│   ├── __init__.py
│   ├── trainer.py           # 主要训练流程实现，支持混合精度、LoRA、知识蒸馏等策略
│   ├── metrics_logger.py    # 记录训练指标到CSV，便于后续结果分析和可视化
│   └── distillation_loss.py # 知识蒸馏损失函数及其封装
├── model/                   # 模型模块目录，定义教师模型和学生模型的结构
│   ├── __init__.py
│   ├── teacher.py           # 定义教师模型（例如预训练的 ResNet-18 模型，并进行微调）
│   └── student.py           # 定义学生模型（例如较小、更浅的 CNN 模型）
├── dataset/                 # 数据集模块目录
│   ├── __init__.py
│   ├── imagenet_loader.py   # 使用 torchvision 加载 ImageNet 数据集的模块
│   ├── data_utils.py        # 通用的数据处理工具函数，如数据划分、数据增强等
│   └── cache/               # 缓存已处理的数据集，避免重复处理
├── utils/                   # 工具函数模块目录
│   ├── __init__.py
│   ├── logger.py            # 日志记录工具，统一设置日志输出（控制台+文件）
│   ├── common.py            # 常用工具函数（例如计算准确率、保存模型等）
│   └── seed.py              # 设置随机数种子，保证实验复现
├── inference/               # 推理模块目录，包含在线推理和批量推理的实现
│   ├── __init__.py
│   └── infer.py             # 推理流程的实现，可加载训练好的模型进行预测
├── analysis/                # 结果分析与可视化模块
│   ├── __init__.py
│   └── analysis.py          # 加载训练日志（CSV）并绘制损失、准确率等曲线
├── data/                    # 原始数据集存储目录
│   └── imagenet/            # 存储 ImageNet 原始数据（训练集和验证集按类别分文件夹）
├── logs/                    # 日志文件存储目录（训练过程中生成的日志文件存放于此）
├── checkpoints/             # 模型检查点存储目录（每个epoch训练结束后保存模型参数）
├── outputs/                 # 输出结果目录（用于存放可视化图表、指标CSV等结果文件）
├── requirements.txt         # 项目依赖说明（包括 PyTorch、TorchVision、PyYAML 等）
└── README.md                # 项目说明文档，详细介绍项目背景、使用方法和环境配置

## 环境配置
- Python 3.12
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```
- 确保 GPU 驱动、CUDA Toolkit 及边缘设备（如 Jetson 系列）正确配置，TensorRT 已安装于边缘设备上。

## 数据集准备
- **CIFAR-10**：请自行下载。
- **ImageNet**：请提前准备好 ImageNet 数据集，并在 `config.yaml` 中设置 `data_dir_imagenet` 路径。目录结构需符合 torchvision 的要求。

## 当前环境介绍
- **Python 3.12.0**
- **torch 版本**: 2.5.1+cu118
- **torchvision 版本**: 0.20.1+cu118
- **pyyaml 版本**: 6.0.1
- **numpy 版本**: 1.26.4
- **scipy 版本**: 1.14.1
- **matplotlib 版本**: 3.9.0

## 运行方式
1. 根据需要在 `config.yaml` 中设置数据集类型（ImageNet）。
2. 运行主程序：
   ```bash
   python main.py
   ```
3. 日志文件将保存在 `logs/` 文件夹中，模型保存于 `checkpoints/`。

## 自动化量化工具
- 推荐使用 [NNCF](https://github.com/openvinotoolkit/nncf) 进行自动化量化策略扩展。
- NVIDIA APEX 可辅助实现混合精度训练，详见 [APEX GitHub](https://github.com/NVIDIA/apex)。


## 参考文献
1. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
2. [MixQuant: Mixed Precision Quantization with a Bit-width Optimization Search](https://arxiv.org/abs/2309.17341)
3. [Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://arxiv.org/abs/1711.05852)
4. [SDQ: Stochastic Differentiable Quantization with Mixed Precision](https://arxiv.org/abs/2206.04459)
5. [HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/abs/1811.08886)
6. [BRECQ: Block Reconstruction-based Quantization for Deep Neural Networks](https://arxiv.org/abs/2004.09576)
