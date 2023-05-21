import torch
from typing import TypedDict


class BatchStep(TypedDict):
    preds: torch.Tensor
    correct_preds: int
    loss: torch.Tensor


class EpochStep(TypedDict):
    accuracy: float
    average_loss: float


class EpochResult(TypedDict):
    training: EpochStep
    testing: EpochStep
