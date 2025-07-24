import wandb
import torch

from tqdm import tqdm
from torch import autocast
from mri.training.metrics import MetricLogger


@torch.inference_mode()
def evaluate_step(
    model,
    batch,
    loss_fn,
    device,
    process_fn,
    metric_logger: MetricLogger
) -> float:
    """
    Evaluate for one step.
    :param model: model to evaluate
    :param batch: input batch
    :param loss_fn: loss function
    :param device: GPU device
    :param process_fn: function to process batch
    :param metric_logger: MetricLogger
    :return: loss value
    """
    baseline_data, followup_data, labels = process_fn(batch, device)
    with autocast(device_type="cuda"):
        outputs = model(baseline_data, followup_data)
        loss = loss_fn(outputs, labels)

    metric_logger.log(y_true=labels.cpu().numpy(), y_pred=outputs.cpu().numpy())

    return loss.item()


@torch.inference_mode()
def evaluate_epoch(
    model,
    loader,
    loss_fn,
    device,
    process_fn,
    metric_logger: MetricLogger,
    scheduler=None,
    wandb_logging=False,
    scheduler_metric=None,
) -> dict:
    """
    Evaluate model.
    :param model: model to evaluate
    :param loader: dataloader
    :param loss_fn: loss function
    :param device: GPU device
    :param process_fn: function to process batch
    :param metric_logger: MetricLogger
    :param scheduler: learning rate scheduler
    :param wandb_logging: flag to log wandb graphs
    :param scheduler_metric: metric to maximize ('f1', 'acc', etc.)
    :return: metrics
    """
    model.eval()
    metric_logger.reset()

    total_loss = 0

    for batch in tqdm(loader, leave=False):
        loss = evaluate_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            device=device,
            process_fn=process_fn,
            metric_logger=metric_logger,
        )

        total_loss += loss

    metrics = metric_logger.get_metrics()
    metrics = {f"valid_{key}": value for key, value in metrics.items()}
    metrics["valid_loss"] = total_loss / len(loader)

    if wandb_logging:
        wandb.log(metrics)

    if scheduler is not None and scheduler_metric is not None:
        scheduler.step(metrics[f"valid_{scheduler_metric}"])

    return metrics
