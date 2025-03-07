# utils/common.py
import torch

def accuracy(outputs, labels):
    # 计算准确率
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total

def save_model(model, path):
    # 保存模型参数
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    # 加载模型参数并移动到设备
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
