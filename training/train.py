import wandb

from tqdm import tqdm
from torch import GradScaler, autocast
from mri.training.metrics import MetricLogger


def train_step(
    step,
    model,
    batch,
    loss_fn,
    scaler,
    optimizer,
    device,
    process_fn,
    metric_logger: MetricLogger,
    batch_accum=1,
) -> float:
    """
    Train for one step.
    :param step: step number
    :param model: model to train
    :param batch: input batch
    :param loss_fn: loss function
    :param scaler: GradScaler
    :param optimizer: optimizer
    :param device: GPU device
    :param process_fn: function to process batch
    :param metric_logger: MetricLogger
    :param batch_accum: batch accumulation steps
    :return: loss value
    """
    baseline_data, followup_data, labels = process_fn(batch, device)
    with autocast(device_type="cuda"):
        outputs = model(baseline_data, followup_data)
        loss = loss_fn(outputs, labels) / batch_accum

    scaler.scale(loss).backward()
    metric_logger.log(y_true=labels.cpu().numpy(), y_pred=outputs.cpu().numpy())

    if step % batch_accum == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item()


def train_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    process_fn,
    metric_logger: MetricLogger,
    scheduler=None,
    wandb_logging=False,
    batch_accum=1,
    scheduler_step="epoch",
) -> dict:
    """
    Train for one epoch.
    :param model: model to train
    :param loader: dataloader
    :param loss_fn: loss function
    :param optimizer: optimizer
    :param device: GPU device
    :param process_fn: function to process batch
    :param metric_logger: MetricLogger
    :param scheduler: learning rate scheduler
    :param wandb_logging: flag to log wandb graphs
    :param batch_accum: batch accumulation steps
    :param scheduler_step: scheduler step frequency ('epoch' or 'batch')
    :return: loss history
    """
    model.train()
    scaler = GradScaler()

    optimizer.zero_grad()
    metric_logger.reset()

    effective_loss = 0
    effective_batch_size = batch_accum * loader.batch_size
    
    train_loss = []

    for batch_idx, batch in enumerate(tqdm(loader, leave=False), start=1):
        loss = train_step(
            step=batch_idx,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            scaler=scaler,
            optimizer=optimizer,
            device=device,
            process_fn=process_fn,
            metric_logger=metric_logger,
            batch_accum=batch_accum,
        )

        effective_loss += loss * effective_batch_size

        if batch_idx % batch_accum == 0:
            metrics = metric_logger.get_metrics()
            metrics = {f"train_{key}": value for key, value in metrics.items()}
            metrics["train_loss"] = effective_loss / effective_batch_size

            effective_loss = 0
            metric_logger.reset()
            train_loss.append(metrics["train_loss"])

            if wandb_logging:
                wandb.log(metrics)

            if scheduler is not None and scheduler_step == "batch":
                scheduler.step()

    if scheduler is not None and scheduler_step == "epoch":
        scheduler.step()
    
    return train_loss
