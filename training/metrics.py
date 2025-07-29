import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn.functional as F

from mri.utils import CLASS_TO_RESPONSE, RESPONSE_TO_CLASS
from sklearn.metrics import roc_auc_score, average_precision_score


def get_tp_fp_fn_tn(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    tn = np.sum((1 - y_true) * (1 - y_pred))
    return tp, fp, fn, tn


class MetricLogger:
    def __init__(self, num_classes: int = 4, logits: bool = False):
        self.reset()
        self.num_classes = num_classes
        self.logits = logits

    def reset(self) -> None:
        self.y_true = []
        self.y_pred = []

    def log(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.y_true.append(y_true)
        if self.logits:
            y_pred_tensor = torch.from_numpy(y_pred)
            y_pred_proba = F.softmax(y_pred_tensor, dim=1).numpy()
            self.y_pred.append(y_pred_proba)
        else:
            self.y_pred.append(y_pred)

    def get_metrics(self) -> dict:
        metrics = {}
        y_true = np.concatenate(self.y_true, axis=0)
        y_pred_proba = np.concatenate(self.y_pred, axis=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        for cls in range(self.num_classes):
            cls_true = (y_true == cls).astype(int)
            cls_pred = (y_pred == cls).astype(int)
            cls_pred_prob = y_pred_proba[:, cls]

            tp, fp, fn, tn = get_tp_fp_fn_tn(cls_true, cls_pred)

            acc = (tp + tn) / (tp + fp + tn + fn)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            ap = (
                average_precision_score(cls_true, cls_pred_prob)
                if np.any(cls_true)
                else 0.0
            )
            roc_auc = (
                roc_auc_score(cls_true, cls_pred_prob)
                if len(np.unique(cls_true)) > 1
                else 0.0
            )

            cls_metrics = {
                "acc": acc,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "ap": float(ap),
                "roc-auc": float(roc_auc),
            }

            metrics.update({
                f"{CLASS_TO_RESPONSE[cls]}_{metric}": value
                for metric, value in cls_metrics.items()
            })

        metrics.update({
            f"{metric}_macro": np.mean(
                [metrics[f"{cls}_{metric}"] for cls in RESPONSE_TO_CLASS]
            )
            for metric in ["acc", "f1", "ap", "roc-auc"]
        })

        return metrics


def show_metrics(train_logs: list[int], valid_logs: list[dict], lr_logs: list[float]) -> None:
    class_names = list(RESPONSE_TO_CLASS.keys())
    valid_loss = [log["valid_loss"] for log in valid_logs]

    clear_output(wait=True)
    plt.figure(figsize=(24, 10))

    # TRAIN LOSS
    plt.subplot(2, 4, 1)
    plt.plot(train_logs, label="Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # VALIDATION LOSS
    plt.subplot(2, 4, 2)
    plt.plot(valid_loss, label="Valid Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(True)
    
    # MACRO METRICS
    plt.subplot(2, 4, 3)
    for metric in ["acc", "f1", "ap"]:
        metric_values = [log[f"valid_{metric}_macro"] for log in valid_logs]
        plt.plot(metric_values, label=f"macro {metric}")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Macro Metrics")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    
    # LEARNING RATE
    plt.subplot(2, 4, 4)
    plt.plot(lr_logs, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate")
    plt.grid(True)

    # PER CLASS PRECISION
    plt.subplot(2, 4, 5)
    last_precision_values = [valid_logs[-1][f"valid_{cls}_precision"] for cls in class_names]
    plt.bar(class_names, last_precision_values, color="lightgreen")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Precision Score")
    plt.title("Class-wise Precision (Last Epoch)")
    plt.ylim(0, 1)
    
    # PER CLASS RECALL
    plt.subplot(2, 4, 6)
    last_recall_values = [valid_logs[-1][f"valid_{cls}_recall"] for cls in class_names]
    plt.bar(class_names, last_recall_values, color="#e74c3c")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Recall Score")
    plt.title("Class-wise Recall (Last Epoch)")
    plt.ylim(0, 1)

    # PER CLASS F1
    plt.subplot(2, 4, 7)
    last_f1_values = [valid_logs[-1][f"valid_{cls}_f1"] for cls in class_names]
    plt.bar(class_names, last_f1_values, color="#f39c12")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("Class-wise F1 (Last Epoch)")
    plt.ylim(0, 1)

    # PER CLASS AVERAGE PRECISION
    plt.subplot(2, 4, 8)
    last_ap_values = [valid_logs[-1][f"valid_{cls}_ap"] for cls in class_names]
    plt.bar(class_names, last_ap_values, color="#3498db")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Average Precision")
    plt.title("Class-wise AP (Last Epoch)")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show();
