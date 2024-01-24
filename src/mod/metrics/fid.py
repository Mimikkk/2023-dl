from torchmetrics.image.fid import FrechetInceptionDistance
from torch import Tensor, uint8


def calculate_fid(images_fake: Tensor, images_real: Tensor):
  fid = FrechetInceptionDistance(feature=64)

  if images_fake.max() > 1.0 or images_real.max() > 1.0:
    images_fake = (images_fake * 255).clamp(0, 255).to(uint8)
    images_real = (images_real * 255).clamp(0, 255).to(uint8)

  fid.update(images_fake.to("cpu"), real=False)
  fid.update(images_real.to("cpu"), real=True)
  return fid.compute().item()
