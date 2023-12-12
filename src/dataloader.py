from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class CelebA(Dataset):
    def __init__(
        self, data_path, annotations_path, images_dir, img_transform=transforms.ToTensor()
    ):
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        self.img_transform = img_transform

        self.annotations = pd.read_csv(
            f"{self.data_path}/{self.annotations_path}", sep=" |  ", skiprows=1, engine='python'
        )
        self.images_list = self.annotations.index.values

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        filename = self.images_list[idx]
        img = Image.open(f"{self.data_path}/{self.images_dir}/{filename}").convert("RGB")

        if self.img_transform is not None:
            img = self.img_transform(img)

        return (img, self.annotations.loc[filename].values)
