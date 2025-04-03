from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Transform
from typing import Tuple
from torch import Tensor
from PIL import Image

import torchvision.transforms.v2 as t
import lightning as L
import pandas as pd
import torch
import os


class BIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_folder_abs_path: str,
        images_labels_csv_abs_path: str,
        transform: Transform = None,
    ):
        super().__init__()
        assert os.path.isdir(
            images_folder_abs_path
        ), "The `images_folder_abs_path` provided is neither valid nor it's pointing to a folder."
        assert os.path.exists(
            images_labels_csv_abs_path
        ), "The `images_labels_csv_abs_path` provided isn't valid"

        self.imgs_path = images_folder_abs_path
        self.imgs_df = pd.read_csv(
            images_labels_csv_abs_path, names=["rel_path", "lbl"]
        )
        self.tensorizer = t.ToImage()
        self.transform = transform

    def __len__(self) -> int:
        return self.imgs_df.shape[0]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        rel_img_path, label = self.imgs_df.iloc[index]
        abs_img_path = os.path.join(self.imgs_path, rel_img_path)
        try:
            with Image.open(abs_img_path) as pil_img:
                tensor_img = self.tensorizer(pil_img)
        except Exception as e:
            raise ValueError(
                f"The absolute path ({abs_img_path}) for the current image isn't valid or the image is corrupted.\n{e}"
            )
        if self.transform:
            tensor_img = self.transform(tensor_img)
        return tensor_img, label


class FitDataManager(L.LightningDataModule):
    def __init__(
        self,
        images_folder_abs_path: str,
        images_labels_csv_abs_path: str,
        train_val_test: Tuple[float, float, float] = (80, 10, 10),
        batch_size=32,
    ):
        assert (
            sum(train_val_test) == 1.0
        ), "The percentages of the train/val/test partitions must add up to 1."
        assert batch_size >= 1, "The batch size should be at least 1."
        self.imgs_path = images_folder_abs_path
        self.imgs_csv = images_labels_csv_abs_path
        self.train_val_test = train_val_test
        self.batch_size = batch_size

    def setup(self, stage):
        dtst = BIDataset(
            self.imgs_path,
            self.imgs_csv,
            transform=t.Compose(t.ToDtype(torch.float32, scale=True)),
        )
        train_set, val_set, test_set = random_split(dtst, self.train_val_test)

        if stage == "fit" or stage is None:
            self.train_set = train_set
        if stage == "validate" or stage is None:
            self.val_set = val_set
        if stage == "test" or stage is None:
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
