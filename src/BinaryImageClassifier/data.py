from torch.utils.data import DataLoader, random_split
from torchvision.models.resnet import ResNet18_Weights
from torchvision.transforms.v2 import Transform
from typing import Tuple
from torch import Tensor
from PIL import Image, ImageFile

import torchvision.transforms.v2 as t
import lightning as L
import pandas as pd
import torch
import cv2
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        valid_entries = []
        for idx, row in self.imgs_df.iterrows():
            rel_path = row["rel_path"]
            img_cv = cv2.imread(os.path.join(images_folder_abs_path, rel_path))
            if img_cv is not None:
                valid_entries.append(row)
            else:
                print(f"[WARNING] Corrupted image skipped: {rel_path}")
        self.imgs_df = pd.DataFrame(valid_entries)
        self.tensorizer = t.ToImage()
        self.transform = transform

    def __len__(self) -> int:
        return self.imgs_df.shape[0]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        rel_img_path, label = self.imgs_df.iloc[index]
        abs_img_path = os.path.join(self.imgs_path, rel_img_path)
        try:
            img_cv = cv2.imread(abs_img_path)
            if img_cv is None:
                raise ValueError(f"OpenCV could not read the image.")
            img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_cv_rgb)
            tensor_img = self.tensorizer(pil_img)
        except Exception as e:
            raise ValueError(
                f"The absolute path ({abs_img_path}) for the current image isn't valid or the image is corrupted.\n{e}"
            )
        if self.transform:
            tensor_img = self.transform(tensor_img)
        return tensor_img, torch.tensor(label, dtype=torch.float32)


class FitDataManager(L.LightningDataModule):
    def __init__(
        self,
        images_folder_abs_path: str,
        images_labels_csv_abs_path: str,
        train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size=32,
        num_workers=3,
    ):
        super().__init__()
        assert (
            sum(train_val_test) == 1.0
        ), "The percentages of the train/val/test partitions must add up to 1."
        assert batch_size >= 1, "The batch size should be at least 1."
        self.imgs_path = images_folder_abs_path
        self.imgs_csv = images_labels_csv_abs_path
        self.train_val_test = train_val_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        dtst = BIDataset(
            self.imgs_path,
            self.imgs_csv,
            transform=t.Compose(
                [
                    t.ToDtype(torch.float32, scale=True),
                    ResNet18_Weights.IMAGENET1K_V1.transforms(),
                ]
            ),
        )
        train_set, val_set, test_set = random_split(dtst, self.train_val_test)

        if stage in ("fit", "validate", None):
            self.val_set = val_set
        if stage in ("fit", None):
            self.train_set = train_set
        if stage in ("test", None):
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
