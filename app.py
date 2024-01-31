import pickle
from typing import Optional, Callable

print('Importing streamlit...')
import streamlit as st
print('Importing torch...')
import torch.nn.parallel
import torch.utils.data
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from math import ceil
import torch
from torchvision.transforms import transforms
from src.mod.datasets import CelebA
from src.mod.datasets.utils import create_latent_vectors
from src.mod.models import Classifier, Generator, LatentSpaceManipulator
print('Importing done.')

loss_fns = {
  'KL Divergence Loss': torch.nn.KLDivLoss(),
  'BCE Loss': torch.nn.BCELoss(),
  'MSE Loss': torch.nn.MSELoss(),
  'Huber Loss': torch.nn.HuberLoss(),
  'Hinge loss': torch.nn.HingeEmbeddingLoss(),
  'Smooth L1': torch.nn.SmoothL1Loss(),
}

@st.cache_resource
def load_device() -> torch.device:
  print('Loading device...')
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_classifier(device: torch.device) -> Classifier:
  print('Loading classifier...')
  return Classifier((3, 64, 64), 40, with_weights=torch.load('demo/c.metrics-combined.pt')).to(device)

@st.cache_resource
def load_generator(device: torch.device) -> Generator:
  print('Loading generator...')
  return Generator(100, 64, 3, with_weights=torch.load('demo/g.metrics-combined.pt')).to(device)

@st.cache_resource
def load_dataset() -> CelebA:
  print('Loading dataset...')
  transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  return CelebA(
    dataset_path='resources/datasets/celeba',
    image_directory='images',
    annotations_directory='annotations',
    image_transform=transform
  )

def array_from_torch(tensor: torch.Tensor) -> np.ndarray:
  return np.transpose(vutils.make_grid(tensor, padding=2, normalize=True).cpu(), (1, 2, 0)).numpy()

def main():
  FeatureThreshold = .4
  LatentVectorSize = 100

  device = load_device()
  generator = load_generator(device)
  classifier = load_classifier(device)
  dataset = load_dataset()
  FeatureCount = len(dataset.annotations.columns)

  with st.container():
    def handle_generation():
      vector = create_latent_vectors(1, LatentVectorSize, device)

      generated = generator(vector)
      features = classifier(generated)

      st.session_state['latent'] = vector
      st.session_state['image'] = array_from_torch(generated)
      st.session_state['features'] = features.cpu().detach().numpy()[0]

    if 'image' not in st.session_state: handle_generation()
    st.button("Generate new image", on_click=handle_generation)
    st.image(st.session_state['image'])

    st.write("Identified features:")

    columns = st.columns(4)
    rows = ceil(len(st.session_state['features']) / len(columns))
    for x in range(rows):
      for y, column in enumerate(columns):
        index = x * len(columns) + y
        if index > FeatureCount: break
        column.checkbox(
          dataset.annotations.columns[index],
          value=st.session_state['features'][index] >= FeatureThreshold,
          key=f'{dataset.annotations.columns[index]}',
          disabled=True
        )

  with st.container():
    NoFeature = "-"
    attribute = st.selectbox("Choose an attribute to flip", (NoFeature, *dataset.annotations.columns))
    loss = st.selectbox("Choose a loss function", loss_fns.keys())
    loss_fn = loss_fns[loss]
    steps = st.slider("Number of steps", min_value=100, max_value=10_000, value=1000)
    alpha = st.slider("Alpha", min_value=0.001, max_value=1.0, value=0.5)

    def handle_manipulation(attribute: str, steps: int, alpha: float):
      index = dataset.annotations.columns.get_loc(attribute)
      target_annotations = st.session_state['features']
      target_annotations[index] = 1.0 if target_annotations[index] < FeatureThreshold else .0

      features: np.ndarray
      images: list[np.ndarray] = []
      def record_steps(
          step: int,
          image: torch.Tensor,
          traits: torch.Tensor,
          _: torch.Tensor
      ):
        nonlocal features

        if step % 40 == 0 or step == steps - 1:
          images.append(array_from_torch(image))

        if step != steps - 1:
          features = traits.cpu().detach().numpy()[0]

      manipulator = LatentSpaceManipulator(generator, classifier, device, loss_fn)
      vector = manipulator.manipulate(
        st.session_state['latent'],
        torch.tensor(target_annotations, device=device),
        alpha=alpha,
        steps=steps,
        on_step=record_steps
      )

      figure = plt.figure(figsize=(8, 8))
      plt.axis("off")
      frames = [[plt.imshow(image, animated=True)] for image in images]
      frames = animation.ArtistAnimation(figure, frames, interval=1000, repeat_delay=1000, blit=True)
      frames.save('animation.gif', writer='imagemagick', fps=20)

      st.session_state['latent_manipulated'] = vector
      st.session_state['image_manipulated'] = images[0]
      st.session_state['features_manipulated'] = features

    st.button("Manipulate", on_click=handle_manipulation, args=(attribute, steps, alpha), disabled=attribute == NoFeature)

    if 'image_manipulated' in st.session_state:
      st.image(st.session_state['image_manipulated'])
      st.image('animation.gif')

      st.write("Identified features:")

      columns = st.columns(4)
      rows = ceil(len(st.session_state['features_manipulated']) / len(columns))
      for x in range(rows):
        for y, column in enumerate(columns):
          index = x * len(columns) + y
          if index > FeatureCount: break
          column.checkbox(
            dataset.annotations.columns[index],
            value=st.session_state['features_manipulated'][index] >= FeatureThreshold,
            key=f'{dataset.annotations.columns[index]}_manipulated',
            disabled=True
          )

if __name__ == '__main__':
  main()
