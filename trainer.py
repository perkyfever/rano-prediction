import os
import abc
import torch

from pathlib import Path
from utils import seed_everything
from logging_info import Experiment, PROJECT_NAME


class BaseTrainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        loss_fn,
        device,
        dataloaders,
        exp_name,
        models_path,
        scheduler=None,
    ):
        """
        Base class for training and evaluating models.
        :param config: Configuration parameters for the experiment.
        :param model: The model to be trained.
        :param optimizer: Optimizer for the model.
        :param loss_fn: Loss function for training.
        :param device: Device to run the model on (GPU).
        :param dataloaders: Dictionary containing training and validation dataloaders.
        :param exp_name: Name of the experiment.
        :param models_path: Path to save model checkpoints.
        :param scheduler: Learning rate scheduler (optional).
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.dataloaders = dataloaders
        self.exp_name = exp_name
        self.models_path = models_path

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.train_scheduler = None
            self.valid_scheduler = scheduler
        else:
            self.train_scheduler = scheduler
            self.valid_scheduler = None

        self.model.to(self.device)
        self.scaler = GradScaler()
        self.trained_epochs = 0

    @abc.abstractmethod
    def train_epoch(self, epoch):
        """
        Trains the model for a single epoch.
        :param epoch: The current epoch number.
        :return: A dictionary containing training metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def evaluate_epoch(self, epoch):
        """
        Evaluates the model for a single epoch.
        :param epoch: The current epoch number.
        :return: A dictionary containing evaluation metrics.
        """
        raise NotImplementedError

    def train(self, num_epochs):
        try:
            for epoch in range(num_epochs):
                self.train_epoch(epoch)
                self.evaluate_epoch(epoch)
                self.trained_epochs += 1
        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            self.save_checkpoint()

    def calculate_metrics(self, predictions, labels):
        """
        Calculates metrics based on predictions and labels.
        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :return: A dictionary of calculated metrics.
        """
        raise NotImplementedError
    
    def monitor_performance(self, )

    def save_checkpoint(self, best_model=False):
        """
        Saves the current model state and optimizer state to a checkpoint file.
        :param best_model: If True, saves the model as the best model.
        """
        checkpoint_dir = self.models_path / self.exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = (
            checkpoint_dir / "best_checkpoint.pth"
            if best_model
            else f"checkpoint_epoch_{self.trained_epochs}.pth"
        )
        torch.save(
            {
                "epoch": self.trained_epochs,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )

    @torch.inference_mode()
    def inference(self, dataloader):
        """
        Runs inference on the provided dataloader.
        :param dataloader: The dataloader for inference.
        :return: Predictions and labels.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.
        :param norm_type: The type of norm to compute (default is 2).
        :return: The computed gradient norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()
