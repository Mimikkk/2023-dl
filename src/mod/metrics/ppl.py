import torch
from torch import nn


def calculate_ppl(images: torch.Tensor) -> float:
  differences = images[:-1] - images[1:]
  path_lengths = torch.sqrt(torch.sum(differences ** 2, dim=[1, 2, 3]))
  return torch.mean(path_lengths).item()
