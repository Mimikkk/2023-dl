import os

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class CelebA(Dataset):
  def __init__(
      self,
      dataset_path: str,
      image_directory: str,
      annotations_directory: str,
      max_image_count: int = None,
      image_transform=transforms.ToTensor(),
  ):
    self.dataset_path = dataset_path
    self.image_directory = image_directory
    self.annotations_directory = annotations_directory
    self.transform = image_transform

    image_count = len(os.listdir(f"{self.dataset_path}/{self.image_directory}"))
    self.size = image_count if max_image_count is None else min(max(0, max_image_count), image_count)

    self.annotations = pd.read_csv(
      f"{self.dataset_path}/{self.annotations_directory}/list_attr_celeba.txt",
      skiprows=1,
      nrows=self.size,
      sep='\s+'
    )

  def __len__(self):
    return self.size

  def __getitem__(self, i: int) -> tuple[Tensor, np.ndarray, str]:
    filename = f'{i + 1:0>6}.jpg'

    image = Image.open(f"{self.dataset_path}/{self.image_directory}/{filename}").convert("RGB")
    image = self.transform(image)

    attributes = self.annotations[self.annotations.index == filename]
    attributes = attributes.replace(-1, 0)
    attributes = attributes.to_numpy().astype(np.float32)
    attributes = torch.from_numpy(attributes)

    return (image, attributes, filename)
