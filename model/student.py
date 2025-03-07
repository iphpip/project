# model/student.py
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        # 定义简单的CNN，作为学生模型，参数量远小于教师模型
        num_classes = config["model"]["num_classes"]
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 对于ImageNet，输入为224×224，经过两次2×2池化后尺寸为56×56
        input_size = 56
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_size * input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_student_model(config):
    # 返回学生模型实例，配置中可扩展其他架构
    student = SimpleCNN(config)
    return student