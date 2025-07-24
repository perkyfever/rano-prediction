import os
import torch
import numpy as np

from pathlib import Path

MODELS_PATH = Path(os.path.abspath(__file__)).parent / "saved_runs"


def load_run_checkpoint(run_name: str):
    """
    Loads run checkpoint.
    :param run_name: str
    :return: checkpoint in a form (config, model, best model)
    """
    checkpoint_path = MODELS_PATH / run_name

    config_checkpoint = None
    config_path = checkpoint_path / "config.pth"
    if os.path.exists(config_path):
        config_checkpoint = torch.load(config_path, weights_only=False)

    model_checkpoint = None
    model_path = checkpoint_path / "model.pth"
    if os.path.exists(model_path):
        model_checkpoint = torch.load(model_path, weights_only=False)

    best_model_checkpoint = None
    best_model_path = checkpoint_path / "best_model.pth"
    if os.path.exists(best_model_path):
        best_model_checkpoint = torch.load(best_model_path, weights_only=False)

    return config_checkpoint, model_checkpoint, best_model_checkpoint


class ModelCheckpointer:
    def __init__(self, run_name: str, checkpoint_freq: int, config: dict | None = None):
        """
        Creates model checkpoints.
        :param run_name: run name
        :param checkpoint_freq: checkpoint saving frequency (in epochs)
        :param metric: early stopping metric
        :param config: additional config to save
        """
        self.epochs_cnt = 0
        self.run_name = run_name
        self.checkpoint_freq = checkpoint_freq
        self.best_metric = -np.inf

        try:
            self.save_path = MODELS_PATH / run_name
            os.makedirs(self.save_path, exist_ok=False)
        except OSError as err:
            print(f"Run with such name already exists. Error: {err}")

        if config is not None:
            config_path = self.save_path / "config.pth"
            torch.save(config, config_path)

    def update(self, model, optimizer, metric=None, scheduler=None, force_checkpoint: bool = False) -> None:
        self.epochs_cnt += 1

        if force_checkpoint or self.epochs_cnt % self.checkpoint_freq == 0:
            self.make_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, best_model=False
            )

        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            self.make_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, best_model=True
            )

    def make_checkpoint(self, model, optimizer, scheduler=None, best_model: bool = False) -> None:
        checkpoint_name = "best_model.pth" if best_model else "model.pth"
        checkpoint_path = self.save_path / checkpoint_name
        checkpoint_data = {
            "epoch": self.epochs_cnt,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        }

        if best_model:
            checkpoint_data.update({"best_metric": self.best_metric})

        torch.save(checkpoint_data, checkpoint_path)
