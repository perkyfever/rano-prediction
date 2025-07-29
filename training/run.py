import torch
import wandb

from tqdm import tqdm

from mri.utils import seed_everything
from mri.models.checkpoint import ModelCheckpointer

from mri.training.metrics import MetricLogger, show_metrics
from mri.training.train import train_epoch
from mri.training.validate import evaluate_epoch

def run_training(
    run_name,
    epochs,
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    device,
    process_fn,
    metric_logger: MetricLogger,
    batch_accum=1,
    wandb_logging=False,
    wandb_config=None,
    scheduler=None,
    scheduler_step="epoch",
    scheduler_metric=None,
    make_checkpoints=True,
    checkpoint_freq=None,
    checkpoint_config=None,
    checkpoint_metric=None,
    overwrite=False,
) -> None:
    """
    Train model.
    :param run_name: run name
    :param epochs: number of epochs to train
    :param model: model to train
    :param train_loader: train loader
    :param valid_loader: validation loader
    :param loss_fn: loss function
    :param optimizer: optimizer
    :param device: GPU device
    :param process_fn: function to process batch
    :param metric_logger: MetricLogger
    :param batch_accum: batch accumulation steps
    :param wandb_logging: flag to log wandb graphs
    :param wandb_config: wandb run config
    :param scheduler: learning rate scheduler
    :param scheduler_metric: metric for scheduler in case it is needed
    :param scheduler_step: scheduler step frequency ('epoch' or 'batch')
    :param make_checkpoints: flag to make checkpoints
    :param checkpoint_freq: checkpoint frequency (in epochs)
    :param checkpoint_config: config to checkpoint
    :param checkpoint_metric: metric for early stopping (e.g. 'f1', 'acc', etc.)
    :param overwrite: if to overwrite the experiment logs
    :return: train loss and validation metrics history
    """
    
    # SOME SANITY CHECKS
    if checkpoint_metric is not None:
        assert checkpoint_metric in ["acc_macro", "f1_macro", "ap_macro", "roc-auc_macro"]
    
    seed_everything()
    model = model.to(device)
    
    checkpointer = None
    if make_checkpoints:
        checkpointer = ModelCheckpointer(
            run_name=run_name,
            checkpoint_freq=checkpoint_freq,
            config=checkpoint_config,
            overwrite=overwrite
        )

    if wandb_logging:
        wandb.init(
            project="MRI Classification",
            name=run_name,
            config=wandb_config
        )

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        train_scheduler = None
        valid_scheduler = scheduler
    else:
        train_scheduler = scheduler
        valid_scheduler = None
    
    train_logs, valid_logs, lr_logs = [], [], []

    try:
        pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:            
            # TRAINING
            train_loss = train_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                process_fn=process_fn,
                metric_logger=metric_logger,
                scheduler=train_scheduler,
                wandb_logging=wandb_logging,
                batch_accum=batch_accum,
                scheduler_step=scheduler_step
            )
            train_logs.extend(train_loss)
            
            # EVALUATING
            metrics = evaluate_epoch(
                model=model,
                loader=valid_loader,
                loss_fn=loss_fn,
                device=device,
                process_fn=process_fn,
                metric_logger=metric_logger,
                scheduler=valid_scheduler,
                wandb_logging=wandb_logging,
                scheduler_metric=scheduler_metric
            )
            valid_logs.append(metrics)

            # LOGGING
            lr_logs.append(optimizer.param_groups[0]["lr"])
            pbar.set_postfix({metric: value for metric, value in valid_logs[-1].items() if "macro" in metric})
            if wandb_logging:
                wandb.log({"lr": lr_logs[-1]})
            else:
                show_metrics(train_logs=train_logs, valid_logs=valid_logs, lr_logs=lr_logs)
            
            # CHECKPOINTING
            if make_checkpoints:
                metric = metrics.get(f"valid_{checkpoint_metric}")
                checkpointer.update(
                    metric=metric,
                    model=model,
                    optimizer=optimizer,
                    scheduler=train_scheduler if train_scheduler is not None else valid_scheduler
                )
                # SAVE ON LAST EPOCH
                if epoch == epochs:
                    checkpointer.update(
                        model=model,
                        optimizer=optimizer,
                        scheduler=train_scheduler if train_scheduler is not None else valid_scheduler,
                        force_checkpoint=True
                    )
        
    except KeyboardInterrupt:
        if make_checkpoints:
            checkpointer.update(
                model=model,
                optimizer=optimizer,
                scheduler=train_scheduler if train_scheduler is not None else valid_scheduler,
                force_checkpoint=True
            )
    finally:
        if wandb_logging:
            wandb.finish()
    
    return train_logs, valid_logs
