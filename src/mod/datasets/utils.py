import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

def split_dataset(dataset: Dataset, split_percent: float, *, use_shuffle: bool = True) -> tuple[DataLoader, DataLoader]:
  train_size = int(split_percent * len(dataset))
  validation_size = len(dataset) - train_size

  (train, validation) = random_split(dataset, [train_size, validation_size])

  train = DataLoader(train, batch_size=32, num_workers=2, shuffle=use_shuffle)
  validation = DataLoader(validation, batch_size=16, num_workers=2, shuffle=use_shuffle)

  return train, validation

def create_latent_vectors(size: int, channel_count: int, device: torch.device) -> Tensor:
  return torch.randn(size, channel_count, 1, 1, device=device, requires_grad=True)
