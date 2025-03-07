# model/teacher.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

def get_teacher_model(config):
    # 使用预训练的 ResNet-18 作为教师模型，并根据任务类别数修改最后一层
    teacher = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_classes = config["model"]["num_classes"]
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    return teacher