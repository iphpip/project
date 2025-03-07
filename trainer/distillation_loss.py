# trainer/distillation_loss.py
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss:
    def __init__(self, temperature=4.0, alpha=0.7):
        self.temperature = temperature
        self.alpha = alpha
        # 学生分类损失（交叉熵）
        self.student_loss_fn = nn.CrossEntropyLoss()
        # 蒸馏损失（KL散度）
        self.kld_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    def __call__(self, student_logits, student_target_loss, teacher_logits):
        # 计算教师与学生输出的蒸馏损失
        distillation_loss = self.kld_loss_fn(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        # 综合损失：分类损失与蒸馏损失加权求和
        loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return loss
