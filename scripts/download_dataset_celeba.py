from scripts.utils import cprint

def iscream(text: str, color: str = None) -> None:
  cprint('download:celeba', text, color, prefix_color='blue')

iscream('preparing torch...')
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets

if __name__ == '__main__':
  iscream("Downloading dataset...")

  data = datasets.CelebA(
    root="resources/datasets",
    split="all",
    download=True,
    transform=ToTensor()
  )

  iscream("Download complete.")
