from torchmetrics.image.fid import FrechetInceptionDistance
import torch


def calculate_fid(images_fake: torch.Tensor, images_real: torch.Tensor):
  fid = FrechetInceptionDistance(feature=64)

  fake_images = (images_fake * 255).clamp(0, 255).to(torch.uint8).to("cpu")
  real_images = (images_real * 255).clamp(0, 255).to(torch.uint8).to("cpu")
  fid.update(fake_images, real=False)
  fid.update(real_images, real=True)
  return fid.compute().item()
