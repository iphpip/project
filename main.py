#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口，根据配置加载数据、模型，并启动训练或推理。
"""

import os
import yaml
import torch
from utils.seed import set_seed
from utils.logger import setup_logger
from dataset import get_dataloaders  # 根据数据集类型加载相应的数据集
from model.teacher import get_teacher_model
from model.student import get_student_model
from trainer.trainer import Trainer

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置文件
    config = load_config()
    
    # 设置随机种子，确保实验结果可复现
    set_seed(42)
    
    # 初始化日志记录器
    logger = setup_logger(config["experiment"]["name"])
    logger.info("配置加载完成")
    
    # 加载数据集，根据配置中的 dataset.name (imagenet) 选择加载方式
    dataset_name = config["dataset"]["name"].lower()
    dataloaders = get_dataloaders(config, dataset_name)
    logger.info(f"{dataset_name} 数据加载完成")

    # 加载教师模型和学生模型
    teacher_model = get_teacher_model(config)
    student_model = get_student_model(config)
    logger.info("教师模型和学生模型加载完成")
    
    # 将模型移动到配置指定的设备上
    device = torch.device(config["experiment"]["device"])
    teacher_model.to(device)
    student_model.to(device)
    
    # 创建训练器实例，并启动训练流程
    trainer = Trainer(teacher_model, student_model, dataloaders, config, logger)
    trainer.train()

if __name__ == "__main__":
    main()