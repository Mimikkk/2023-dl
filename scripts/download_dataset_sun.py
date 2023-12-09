import os
from utils import cprint, chalk
import wget
import tarfile
import shutil

def iscream(text: str, color: str = None) -> None:
  cprint('download:sun', text, color, prefix_color='blue')

if __name__ == '__main__':
  directory = 'resources/datasets/sun'
  cache = f'{directory}/cache'
  annotations_name = 'SUNAttributeDB.tar.gz'
  annotations_url = f"https://cs.brown.edu/~gmpatter/Attributes/{annotations_name}"
  annotations_dir = f'{directory}/annotations'
  annotations_file = f'{cache}/annotations.tar.gz'
  images_name = 'SUNAttributeDB_Images.tar.gz'
  images_url = f"https://cs.brown.edu/~gmpatter/Attributes/{images_name}"
  images_dir = f'{directory}/images'
  images_file = f'{cache}/images.tar.gz'

  from pathlib import Path
  def up_one_dir(path):
    try:
      parent = Path(path).parents[0]
      parent_parent = Path(path).parents[1]

      os.rename(path, f"{parent_parent}/temp")
      os.rename(f"{parent_parent}/temp", parent)
    except IndexError: ...


  iscream('Preparing downloads...')

  if os.getcwd().endswith('scripts'): os.chdir('..')

  def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
      os.makedirs(path)

  def is_cached(path: str) -> bool:
    return os.path.exists(path)

  iscream("- Downloading dataset...")

  if is_cached(annotations_file):
    iscream('-- Using cached annotations.')
  else:
    iscream('-- Downloading annotations...')
    iscream(f"- From {annotations_url}")
    ensure_dir(cache)
    wget.download(annotations_url, cache)
    shutil.move(f'{cache}/{annotations_name}', annotations_file)

  if is_cached(images_file):
    iscream('-- Using cached images.')
  else:
    iscream('-- Downloading images...')
    iscream(f"-- From {images_url}")
    ensure_dir(cache)
    wget.download(images_url, cache)
    shutil.move(f'{cache}/{images_name}', images_file)

  iscream('Downloads complete.')
  def is_extracted(path: str) -> bool:
    return os.path.exists(path)


  def track_progress(members):
    extracted = 0
    iscream('-- Calculating total...')
    total = len(members.getnames())

    for member in members:
      yield member
      extracted += 1

      if extracted % 100 == 0 or extracted == total:
        current = chalk(f'({extracted}/{total})', 'yellow')
        iscream(f'-- Extracting {current} - {member.name}', 'green')

  iscream('Preparing extraction...')
  if is_extracted(annotations_dir):
    iscream('- Annotations already extracted.')
  else:
    iscream('- Extracting annotations...')
    ensure_dir(annotations_dir)
    with tarfile.open(annotations_file, 'r:gz') as tar:
      tar.extractall(annotations_dir, members=track_progress(tar))
      up_one_dir(f'{annotations_dir}/SUNAttributeDB')

  if is_extracted(images_dir):
    iscream('- Images already extracted.')
  else:
    iscream('- Extracting images...')
    ensure_dir(images_dir)
    with tarfile.open(images_file, 'r:gz') as tar:
      tar.extractall(images_dir, members=track_progress(tar))
      up_one_dir(f'{images_dir}/images/')

  iscream('Extraction complete.')
  iscream('Dataset ready.')
