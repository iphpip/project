# trainer/metrics_logger.py
import csv
import os

class MetricsLogger:
    def __init__(self, filename="metrics.csv"):
        self.filename = filename
        self.fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writeheader()

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
