from utils import cprint, ensure_cwd

def iscream(text: str, color: str = None) -> None:
  cprint('download:celeba', text, color, prefix_color='blue')

iscream('preparing torch...')
import torchvision.datasets as datasets

def main():
  iscream("Downloading dataset...")

  datasets.CelebA(
    root="resources/datasets",
    split="all",
    download=True,
  )

  iscream("Download complete.")

if __name__ == '__main__':
  ensure_cwd()
  main()
