from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torchvision.utils as vutils

def show_images(*images: Tensor):
  if len(images) > 1:
    _, axes = plt.subplots(1, len(images), figsize=(16, 8))
    for i, image in enumerate(images):
      grid = vutils.make_grid(image, padding=2, normalize=True)
      axes[i].axis('off')
      axes[i].imshow(np.transpose(grid.cpu().detach().numpy(), (1, 2, 0)))
  else:
    grid = vutils.make_grid(images[0], padding=2, normalize=True)
    plt.imshow(np.transpose(grid.cpu().detach().numpy(), (1, 2, 0)))

  plt.axis('off')
  plt.show()
