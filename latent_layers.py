import os
import torch
import matplotlib.pyplot as plt

from demo.utils.utils import show_images
from src.mod.datasets.utils import create_latent_vectors
from src.mod.models import Generator


def ensure_directory(path: str):
  if not os.path.exists(path): os.makedirs(path)


def main():
  LatentVectorSize = 100
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = Generator(
    LatentVectorSize,
    64,
    3,
    with_weights=torch.load('demo/g.run_three.pt')
  ).eval().to(device)

  outputs: list[tuple[str, torch.Tensor]] = []
  subscribers = []
  add_layer = lambda _, __, out: outputs.append((f'{len(outputs)}-conv2T-layer', out))
  for output in model.modules():
    if isinstance(output, torch.nn.modules.conv.ConvTranspose2d):
      subscribers.append(output.register_forward_hook(add_layer))

  def visualize_feature_maps(output: tuple[str, torch.Tensor], columns: int = 6):
    (label, tensor) = output
    tensor = tensor.cpu().detach()

    kernel_count = tensor.shape[1]
    rows = 1 + kernel_count // columns
    figure = plt.figure(figsize=(columns, rows))
    for i in range(kernel_count):
      axis = figure.add_subplot(rows, columns, i + 1)
      axis.imshow(tensor[0, i, :, :], cmap='gray')
      axis.axis('off')
      axis.set_xticklabels([])
      axis.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f'./resources/figures-latent/{label}.png')

  ensure_directory('./resources/figures-latent/')

  vectors = create_latent_vectors(1, LatentVectorSize, device)
  generated = model(vectors)
  show_images(generated)
  plt.savefig('./resources/figures-latent/latent_layers-image.png')

  for output in outputs:
    visualize_feature_maps(output)

  for subscriber in subscribers:
    subscriber.remove()


if __name__ == '__main__':
  main()
