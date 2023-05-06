import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .common.types import EpochStep

from .model import Model
from .common.constants import CUDA_IS_AVAILABLE
from torch.utils.data import DataLoader
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Flatten(),
                nn.Sequential(
                    nn.Linear(64 * 4 * 4, 10),
                ),
            ]
        )

    def forward(self, x: torch.Tensor, target):
        for layer in self.layers:
            x = layer(x)

        return x


class Cifar10Model(Model):
    model: Net
    optimizer: optim.SGD
    loss_criterion: nn.CrossEntropyLoss

    def __init__(self, path: str) -> None:
        super().__init__()
        self.model = Net()
        if CUDA_IS_AVAILABLE:
            self.model.cuda()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
        )
        self.loss_criterion = nn.CrossEntropyLoss()
        self.path = path

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ) -> None:
        best_accuracy: float = -np.inf
        for epoch in range(epochs):
            print(f"\nN° Epoch {epoch}")
            self.model.train()
            train_epoch_step = self.epoch_step(train_loader, train=True)
            print(
                f"Trainning loss: {train_epoch_step['average_loss']}\tAccuracy: {train_epoch_step['accuracy']}"
            )

            if test_loader:
                self.model.eval()
                test_epoch_step = self.epoch_step(test_loader, train=False)
                print(
                    f"Test loss: {test_epoch_step['average_loss']}\tAccuracy: {test_epoch_step['accuracy']}"
                )
                if best_accuracy is None or best_accuracy < test_epoch_step["accuracy"]:
                    print(
                        f"Best Accuracy Found: {best_accuracy} -> {test_epoch_step['accuracy']}. Saving Model..."
                    )
                    best_accuracy = test_epoch_step["accuracy"]
                    self.save_weights()

    def epoch_step(self, dataset_loader: DataLoader, train=False) -> EpochStep:
        total_correct_preds, total_loss = 0, 0

        sample_count = 0
        for x, target in dataset_loader:
            if train:
                self.optimizer.zero_grad()

            x, target = Variable(x), Variable(target)
            if CUDA_IS_AVAILABLE:
                x, target = x.cuda(), target.cuda()

            sample_count += len(x)

            score = self.model(x, target)
            loss = self.loss_criterion(score, target)

            _, pred_label = torch.max(score.data, 1)
            correct_preds = (pred_label == target.data).sum()
            total_correct_preds += correct_preds
            total_loss += loss.item()
            if train:
                loss.backward()
                self.optimizer.step()

        accuracy = total_correct_preds / sample_count
        average_loss = total_loss / sample_count

        return {"accuracy": accuracy, "average_loss": average_loss}

    def evaluate(self, validation_loader: DataLoader):
        validation_epoch_step = self.epoch_step(validation_loader, train=False)
        print(
            f"\nValidation loss: {validation_epoch_step['average_loss']}\tAccuracy: {validation_epoch_step['accuracy']}"
        )
