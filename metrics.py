from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from src.mod.datasets import CelebA
from src.mod.datasets.utils import create_latent_vectors
from src.mod.metrics import calculate_fid, calculate_ppl
from src.mod.models import Generator


def main():
  LatentVectorSize = 100
  EvaluationSampleCount = 1000

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  generator = Generator(
    LatentVectorSize,
    64,
    3,
    with_weights=torch.load('demo/g.run_three.pt')
  ).eval().to(device)

  image_size = 64
  transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
  ])

  dataloader = DataLoader(
    CelebA(
      dataset_path='resources/datasets/celeba',
      image_directory='images',
      annotations_directory='annotations',
      image_transform=transform
    ),
    batch_size=EvaluationSampleCount,
    shuffle=True
  )

  vectors = create_latent_vectors(
    EvaluationSampleCount,
    LatentVectorSize,
    device
  )
  fake_images = generator(vectors)
  real_images, _, _ = next(iter(dataloader))

  fid = calculate_fid(fake_images, real_images)
  ppl = calculate_ppl(generator, device, LatentVectorSize, EvaluationSampleCount)
  print(f'FID Score: {fid:.3f} ( Lower is better, InputSize dependent )')
  print(f'PPL Score: {ppl:.3f} ( Lower Usually is better as its smoother )')


if __name__ == '__main__':
  main()
