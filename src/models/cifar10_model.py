import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from .common.types import EpochStep, EpochResult, BatchStep

from .model import Model
from .common.constants import CUDA_IS_AVAILABLE
from .res_net_18 import ResNet18_32x32
from .simple_net import SimpleNet_32x32
from torch.utils.data import DataLoader
from torch.autograd import Variable

from typing import Type


class Cifar10Model(Model):
    model: nn.Module
    optimizer: optim.SGD
    loss_criterion: nn.CrossEntropyLoss

    def __init__(
        self, path: str, model: Type[ResNet18_32x32 | SimpleNet_32x32]
    ) -> None:
        super().__init__()

        self.model = model(num_outputs=10)
        if CUDA_IS_AVAILABLE:
            self.model.cuda()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.loss_criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.path = path

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> list[EpochResult]:
        best_accuracy: float = -np.inf
        history: list[EpochResult] = []

        for epoch in range(epochs):
            print(f"\nN° Epoch {epoch}")
            train_epoch_step = self.epoch_step(train_loader, train=True)
            print(
                f"Trainning loss: {train_epoch_step['average_loss']}\tAccuracy: {train_epoch_step['accuracy']}"
            )

            test_epoch_step = self.epoch_step(test_loader, train=False)
            history.append(
                {
                    "training": train_epoch_step,
                    "testing": test_epoch_step,
                }
            )
            print(
                f"Test loss: {test_epoch_step['average_loss']}\tAccuracy: {test_epoch_step['accuracy']}"
            )
            if best_accuracy is None or best_accuracy < test_epoch_step["accuracy"]:
                print(
                    f"Better Accuracy Found: {best_accuracy} -> {test_epoch_step['accuracy']}. Saving Model..."
                )
                best_accuracy = test_epoch_step["accuracy"]
                self.save_weights()

            self.scheduler.step()
        return history

    def evaluate(self, validation_loader: DataLoader) -> EpochStep:
        validation_epoch_step = self.epoch_step(validation_loader, train=False)
        print(
            f"\nValidation loss: {validation_epoch_step['average_loss']}\tAccuracy: {validation_epoch_step['accuracy']}"
        )
        return validation_epoch_step

    def epoch_step(self, dataset_loader: DataLoader, train=False) -> EpochStep:
        total_correct_preds, total_loss = 0, 0

        sample_count = 0

        if train:
            self.model.train()
        else:
            self.model.eval()

        for batch in dataset_loader:
            if train:
                self.optimizer.zero_grad()
            sample_count += len(batch[0])
            batch_step = self.batch_step(batch)

            total_correct_preds += batch_step["correct_preds"]
            total_loss += batch_step["loss"].item()
            if train:
                batch_step["loss"].backward()
                self.optimizer.step()

        accuracy = total_correct_preds / sample_count
        average_loss = total_loss / sample_count

        return {"accuracy": accuracy, "average_loss": average_loss}

    def batch_step(self, batch) -> BatchStep:
        x, target = batch
        x, target = Variable(x), Variable(target)
        if CUDA_IS_AVAILABLE:
            x, target = x.cuda(), target.cuda()

        score = self.model(x, target)
        loss = self.loss_criterion(score, target)

        _, pred_label = torch.max(score.data, 1)
        correct_preds = int((pred_label == target.data).sum())

        return {"preds": pred_label, "correct_preds": correct_preds, "loss": loss}
