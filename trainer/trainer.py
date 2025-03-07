# trainer/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from trainer.metrics_logger import MetricsLogger
from trainer.distillation_loss import DistillationLoss
from utils.common import accuracy, save_model
import math
import os

class Trainer:
    def __init__(self, teacher_model, student_model, dataloaders, config, logger):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.config = config
        self.logger = logger
        self.device = torch.device(config["experiment"]["device"])

        # 是否冻结教师模型，默认冻结用于知识蒸馏
        teacher_update = config.get("teacher_update", False)
        if not teacher_update:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.logger.info("教师模型设置为可更新（微调）模式。")
            self.teacher_model.train()

        # 混合精度训练设置
        self.use_amp = config["mixed_precision"].get("use_amp", False)
        if self.use_amp:
            self.scaler = GradScaler()

        # 优化器只更新学生模型参数（如果教师模型参与更新，则需要额外处理）
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=config["experiment"]["learning_rate"],
            weight_decay=float(config["experiment"]["weight_decay"])
        )

        # 计算训练总步数及梯度累计步数
        self.gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        self.epochs = config["experiment"]["epochs"]
        self.total_steps = self.epochs * self.num_update_steps_per_epoch

        # 这里以StepLR为例，您也可以选择其他调度策略
        step_size = max(1, self.total_steps // 10)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)

        # 初始化知识蒸馏损失函数（融合分类损失和教师指导的KL散度损失）
        self.distill_loss_fn = DistillationLoss(
            temperature=config["distillation"]["temperature"],
            alpha=config["distillation"]["alpha"]
        )

        # 初始化指标记录器，保存训练与验证日志
        self.metrics_logger = MetricsLogger(filename=os.path.join(config["logging"]["logging_dir"], "metrics.csv"))

    def train(self):
        global_step = 0
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.epochs} 开始训练")
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)
            self.metrics_logger.log(epoch, train_loss, train_acc, val_loss, val_acc)
            self.logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, "
                             f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
            self.lr_scheduler.step()

            # 每个epoch结束后可选择保存模型
            model_save_path = os.path.join("checkpoints", f"{self.config['experiment']['name']}_epoch{epoch}.pth")
            os.makedirs("checkpoints", exist_ok=True)
            save_model(self.student_model, model_save_path)
            self.logger.info(f"学生模型保存至 {model_save_path}")
            global_step += self.num_update_steps_per_epoch

    def _train_one_epoch(self, epoch):
        self.student_model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        self.optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.student_model = self.student_model.to(self.device)
            self.teacher_model = self.teacher_model.to(self.device)

            # 前向计算采用混合精度（若启用）
            if self.use_amp:
                # 添加 device_type 参数
                with autocast(device_type=self.device.type):
                    student_logits = self.student_model(inputs)
                    student_loss = nn.CrossEntropyLoss()(student_logits, labels)
                    # 使用教师模型计算软标签输出（不更新教师参数）
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(inputs)
                    loss = self.distill_loss_fn(student_logits, student_loss, teacher_logits)
                self.scaler.scale(loss).backward()
            else:
                student_logits = self.student_model(inputs)
                student_loss = nn.CrossEntropyLoss()(student_logits, labels)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(inputs)
                loss = self.distill_loss_fn(student_logits, student_loss, teacher_logits)
                loss.backward()

            # 梯度累计
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            self.optimizer.zero_grad()
            #self.lr_scheduler.step() 

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(student_logits, dim=1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

            if (batch_idx + 1) % self.config["logging"]["log_interval"] == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        self.lr_scheduler.step()
        epoch_loss = running_loss / total_samples
        epoch_acc = (running_corrects / total_samples) * 100.0
        return epoch_loss, epoch_acc

    def _validate(self, epoch):
        self.student_model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                student_logits = self.student_model(inputs)
                loss = nn.CrossEntropyLoss()(student_logits, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(student_logits, dim=1)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (running_corrects / total_samples) * 100.0
        return epoch_loss, epoch_acc
