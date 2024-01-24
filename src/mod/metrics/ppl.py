from torch import Tensor, sqrt, sum, mean


def calculate_ppl(images: Tensor) -> float:
  return mean(sqrt(sum((images[:-1] - images[1:]) ** 2, dim=[1, 2, 3]))).item()
