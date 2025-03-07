# inference/infer.py
import torch
import argparse
from model.student import get_student_model
from utils.common import load_model
from PIL import Image
import torchvision.transforms as transforms
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的学生模型参数路径")
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def infer():
    args = parse_args()
    device = torch.device(args.device)
    # 构建简单CNN学生模型，注意与训练时配置保持一致
    config = load_config()
    model = get_student_model(config)
    model = load_model(model, args.model_path, device)
    model.eval()

    # 图像预处理：保持与训练时一致
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = Image.open(args.image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(f"预测类别: {predicted.item()}")

if __name__ == "__main__":
    infer()