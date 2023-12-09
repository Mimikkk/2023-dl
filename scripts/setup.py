import subprocess
import sys
from utils import in_venv, cprint, ensure_cwd

def install(*packages: list[str]) -> None:
  subprocess.call([sys.executable, "-m", "pip", "install", *packages])

def iscream(text: str, color: str = None) -> None:
  cprint('install', text, color, prefix_color='cyan')

iscream('Checking if in virtual environment...')
if not in_venv():
  iscream('Not in virtual environment. Installing virtualenv...')
  install('virtualenv')

  iscream('Creating virtual environment...')
  subprocess.call([sys.executable, "-m", "virtualenv", "venv"])

  iscream('Activating virtual environment...')
  subprocess.call([sys.executable, "-m", "venv", "venv"])
  iscream('Virtual environment created. Please restart the script.')
  sys.exit(0)

iscream("Inside virtual env.")
iscream('Installing packages...')

iscream('Installing torch. Ensure CUDA is installed...')
install('torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118')

iscream('Other dependencies...')
install('requests', 'pandas', 'numpy', 'icecream', 'wget')

iscream('Installation successful.')
