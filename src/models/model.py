import torch
import torch.nn as nn
import torch.optim as optim
import os

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from .common.types import EpochStep, EpochResult, BatchStep


class Model(ABC):
    model: nn.Module
    optimizer: optim.Optimizer
    loss_criterion: nn.modules.loss._Loss
    path: str

    @abstractmethod
    def train(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> list[EpochResult]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, validation_loader: DataLoader) -> EpochStep:
        raise NotImplementedError

    @abstractmethod
    def epoch_step(self, dataset_loader: DataLoader, train=False) -> EpochStep:
        raise NotImplementedError

    @abstractmethod
    def batch_step(self, batch) -> BatchStep:
        raise NotImplementedError

    def save_weights(self):
        torch.save(self.model.state_dict(), self.path)

    def load_weights(self):
        if os.path.isfile(self.path):
            self.model.load_state_dict(torch.load(self.path))
