from typing import Optional

import streamlit as st
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

def main():
  ModelsDirectory = "demo/"
  # >= threshold -> feature is treated present
  FeatureThreshold = .4
  NoFeature = "-"
  LatentVectorSize = 100

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  generator = Generator(
    LatentVectorSize,
    64,
    3,
    with_weights=torch.load(ModelsDirectory + 'g.run_three.pt', map_location=torch.device(device))
  )
  classifier = Classifier(
    (3, 64, 64),
    40,
    with_weights=torch.load(ModelsDirectory + 'c.run_three.pt', map_location=torch.device(device))
  )

  dataset = CelebA(
    dataset_path='resources/datasets/celeba',
    image_directory='img_align_celeba',
    annotations_directory='annotations',
    image_transform=transform
  )

  def generate_image(name_postfix: str = '', noise: Optional[torch.Tensor] = None):
    if not name_postfix or noise is None:
      noise = create_latent_vectors(1, LatentVectorSize, device)

    noise.to(device)
    generator.to(device)
    classifier.to(device)

    generated = generator(noise)
    features = classifier(generated).cpu().detach().numpy()[0]

    st.session_state[f'latent{name_postfix}'] = noise
    st.session_state[f'image{name_postfix}'] = np.transpose(vutils.make_grid(generated, padding=2, normalize=True).cpu(), (1, 2, 0)).numpy()
    st.session_state[f'features{name_postfix}'] = features

  def handle_manipulation(attribute: str, steps: int, alpha: float):
    if attribute == NoFeature: return

    attribute_index = list(dataset.annotations.columns).index(choice)
    target_annotations = st.session_state['features'].copy()
    target_annotations[attribute_index] = 1.0 if target_annotations[attribute_index] < FeatureThreshold else .0

    vector, vectors = manipulator.manipulate(
      st.session_state['latent'].clone(),
      torch.tensor(target_annotations, device=device),
      alpha=alpha,
      steps=steps
    )

    figure = plt.figure(figsize=(8, 8))
    plt.axis("off")

    images = [
      [
        plt.imshow(
          np.transpose(
            vutils.make_grid(generator(torch.tensor(i, device=device)).cpu().squeeze().detach(), padding=2, normalize=True),
            (1, 2, 0),
          ),
          animated=True
        )
      ] for i in vectors[::40]
    ]

    print(len(images), 'saving...')
    frames = animation.ArtistAnimation(figure, images, interval=1000, repeat_delay=1000, blit=True)
    frames.save('animation.gif', writer='imagemagick', fps=20)
    print('saved')

    generate_image(name_postfix="_manipulated", noise=vector)


  manipulator = LatentSpaceManipulator(generator, classifier, device)
  base = st.container()
  target = st.container()

  if 'image' not in st.session_state:
    generate_image()
  if 'image_manipulated' not in st.session_state:
    st.session_state['image_manipulated'] = None

  with base:
    st.button("Generate new image", on_click=generate_image)
    st.image(st.session_state['image'])

    st.write("Identified features:")

    columns = st.columns(4)
    rows = ceil(len(st.session_state['features']) / len(columns))

    for x in range(rows):
      for y, column in enumerate(columns):
        index = x * len(columns) + y
        if index < len(dataset.annotations.columns):
          column.checkbox(
            dataset.annotations.columns[index],
            value=st.session_state['features'][index] >= FeatureThreshold
          )

  with target:
    choice = st.selectbox("Choose attribute to flip", [NoFeature, *dataset.annotations.columns])
    steps = st.slider("Number of steps", min_value=100, max_value=10_000, value=1000)
    alpha = st.slider("Alpha", min_value=0.001, max_value=1.0, value=0.1)

    st.button("Manipulate", on_click=handle_manipulation, args=(choice, steps, alpha))

    if st.session_state['image_manipulated'] is not None:
      st.image(st.session_state['image_manipulated'])
      st.image('animation.gif')

      st.write("Identified features:")
      columns = st.columns(3)
      rows = ceil(len(st.session_state['features_manipulated']) / len(columns))

      for x in range(rows):
        for y, column in enumerate(columns):
          index = x * len(columns) + y
          if index < len(dataset.annotations.columns):
            column.checkbox(
              dataset.annotations.columns[index],
              value=st.session_state['features_manipulated'][index] >= FeatureThreshold,
              key=f'{dataset.annotations.columns[index]}_manipulated'
            )


if __name__ == '__main__':
  main()
