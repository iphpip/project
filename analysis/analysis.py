# analysis/analysis.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_file="logs/metrics.csv"):
    # 读取CSV中的训练指标
    df = pd.read_csv(csv_file)
    
    # 绘制Loss曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    # 绘制Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Val Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics()
