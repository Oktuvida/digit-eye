import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int):
    return nn.Conv2d(
        in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int):
    return nn.Conv2d(
        in_channels, out_channels, stride=stride, kernel_size=1, bias=False
    )


def norm_layer(channels: int):
    return nn.BatchNorm2d(channels)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3(in_channels, out_channels, stride=stride),
                    norm_layer(out_channels),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    conv3x3(out_channels, out_channels, stride=1),
                    norm_layer(out_channels),
                ),
            ]
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                norm_layer(out_channels),
            )

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_32x32(nn.Module):
    def __init__(self, num_outputs: int) -> None:
        super().__init__()

        self.in_channels = 64

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3,
                        self.in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    norm_layer(self.in_channels),
                    nn.ReLU(),
                ),
                self.__get_block(64, stride=1),
                self.__get_block(128, stride=2),
                self.__get_block(256, stride=2),
                self.__get_block(512, stride=2),
                nn.AvgPool2d(4),
            ]
        )

        self.linear = nn.Linear(512, num_outputs)

    def __get_block(self, out_channels: int, stride: int):
        strides = [stride, 1]
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
