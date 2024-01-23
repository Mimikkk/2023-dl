import streamlit as st
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from demo.utils.utils import show_images
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils as vutils
from typing import Callable
import os
import PIL.Image as Image
import numpy as np
import pandas as pd
from math import ceil
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Optional
from src.mod.datasets.utils import create_latent_vectors
from src.mod.plugins.early_stopping import EarlyStopping
from src.mod.datasets import CelebA
from src.mod.models import Classifier, Generator, LatentSpaceManipulator
from src.mod.datasets.utils import split_dataset

MODELS_DIR = "demo/"
FEATURE_THRESHOLD = .4  # >= threshold -> feature is treated present
NO_CHOSEN_FEATURE = "-"

latent_vector_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
  transforms.Resize(64),
  transforms.CenterCrop(64),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

generator = Generator(latent_vector_size, 64, 3, with_weights=torch.load(MODELS_DIR + 'g.run_three.pt', map_location=torch.device(device)))
classifier = Classifier((3, 64, 64), 40, with_weights=torch.load(MODELS_DIR + 'c.run_three.pt', map_location=torch.device(device)))
dataset = CelebA(
  dataset_path='resources/datasets/celeba',
  image_directory='images',
  annotations_directory='annotations',
  image_transform=transform
)

def gen_img(variable_postfix: str = '', noise = None):
  if not variable_postfix or noise is None:
    noise = torch.randn(1, latent_vector_size, 1, 1, device=device)

  fake = generator(noise)
  features = classifier(fake).cpu().detach().numpy()[0]

  st.session_state['latent' + variable_postfix] = noise
  st.session_state['img' + variable_postfix] = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)).numpy()
  st.session_state['features' + variable_postfix] = features

def manipulate(attribute_name: str, steps: int, alpha: float):
  if attribute_name == NO_CHOSEN_FEATURE:
     return

  attribute_idx = list(dataset.annotations.columns).index(choice)
  target_annotations = st.session_state['features'].copy()
  target_annotations[attribute_idx] = 1.0 if target_annotations[attribute_idx] < FEATURE_THRESHOLD else .0

  new_latent, vecs = manipulator.manipulate(st.session_state['latent'].clone(), torch.tensor(target_annotations, device=device), alpha=alpha, steps=steps)

  fig = plt.figure(figsize=(8,8))
  plt.axis("off")

  ims = [[
    plt.imshow(
      np.transpose(
        vutils.make_grid(generator(torch.tensor(i, device=device)).squeeze().detach(), padding=2, normalize=True),
        (1,2,0),
      ),
      animated=True
    )] for i in vecs[::40]]

  ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
  ani.save('animation.gif', writer='imagemagick', fps=20)

  gen_img(variable_postfix="_manipulated", noise=new_latent)


manipulator = LatentSpaceManipulator(generator, classifier, device)
base = st.container()
target = st.container()

if 'img' not in st.session_state:
    gen_img()
if 'img_manipulated' not in st.session_state:
    st.session_state['img_manipulated'] = None

with base:
  st.button("Generate new image", on_click=gen_img)
  st.image(st.session_state['img'])

  st.write("Identifies features:")

  cols = st.columns(3)
  rows = ceil(len(st.session_state['features']) / len(cols))

  for i in range(rows):
      for j, col in enumerate(cols):
          idx = i * len(cols) + j
          if idx < len(dataset.annotations.columns):
            col.checkbox(dataset.annotations.columns[idx], value=st.session_state['features'][idx]>=FEATURE_THRESHOLD)

with target:
  choice = st.selectbox("Choose attribute to flip", [NO_CHOSEN_FEATURE] + list(dataset.annotations.columns))
  steps = st.slider("Number of steps", min_value=100, max_value=10_000, value=1000)
  alpha = st.slider("Alpha", min_value=0.001, max_value=1.0, value=0.1)

  st.button("Manipulate", on_click=manipulate, args=(choice, steps, alpha))

  if st.session_state['img_manipulated'] is not None:
    st.image(st.session_state['img_manipulated'])

    st.image('animation.gif')

    st.write("Identifies features:")

    cols = st.columns(3)
    rows = ceil(len(st.session_state['features_manipulated']) / len(cols))


    for i in range(rows):
        for j, col in enumerate(cols):
            idx = i * len(cols) + j
            if idx < len(dataset.annotations.columns):
              col.checkbox(dataset.annotations.columns[idx], value=st.session_state['features_manipulated'][idx]>=FEATURE_THRESHOLD, key=f'{dataset.annotations.columns[idx]}_manipulated')



