import torch
from torch import nn

from src.mod.datasets.utils import create_latent_vectors


def calculate_ppl(
    generator: nn.Module,
    device: torch.device,
    latent_vector_size: int,
    sample_count: int
) -> float:
  vectors = create_latent_vectors(sample_count, latent_vector_size, device)

  images = generator(vectors)
  differences = images[:-1] - images[1:]
  path_lengths = torch.sqrt(torch.sum(differences ** 2, dim=[1, 2, 3]))
  return torch.mean(path_lengths).item()
