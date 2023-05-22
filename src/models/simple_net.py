import torch
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int):
    return nn.Conv2d(
        in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False
    )


class SimpleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ##
                conv3x3(in_channels, out_channels),
                nn.ReLU(),
                ##
                conv3x3(out_channels, out_channels),
                nn.ReLU(),
                ##
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
            ]
        )

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class SimpleNet_32x32(nn.Module):
    def __init__(self, num_outputs: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ## Convolutional layers
                SimpleBlock(3, 32),
                SimpleBlock(32, 64),
                SimpleBlock(64, 128),
                ## Flatten
                nn.Flatten(),
                ## Linear layers
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(),
                ##
                nn.Linear(512, 128),
                nn.ReLU(),
                ##
                nn.Linear(128, num_outputs),
                nn.LogSoftmax(dim=1),
            ]
        )

    def forward(self, x: torch.Tensor, target):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
