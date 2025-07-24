import numpy as np

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
            ap = average_precision_score(cls_true, cls_pred_prob) if np.any(cls_true) else 0.0
            roc_auc = roc_auc_score(cls_true, cls_pred_prob) if len(np.unique(cls_true)) > 1 else 0.0

            cls_metrics = {
                "acc": acc,
                "f1": f1,
                "ap": float(ap),
                "roc-auc": float(roc_auc),
            }

            metrics.update({
                f"{CLASS_TO_RESPONSE[cls]}_{metric}": value
                for metric, value in cls_metrics.items()
            })

        metrics.update({
            f"{metric}_macro": np.mean([metrics[f"{cls}_{metric}"] for cls in RESPONSE_TO_CLASS])
            for metric in ["acc", "f1", "ap", "roc-auc"]
        })

        return metrics
