from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models.resnet import ResNet18_Weights
from torchvision.transforms.v2 import Transform
from PIL import Image, ImageFile
from typing import Tuple
from torch import Tensor

import torchvision.transforms.v2 as t
import lightning as L
import pandas as pd
import torch
import cv2
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Transform = None,
    ):
        super().__init__()
        self.imgs_df = df
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


class Augment(Dataset):
    def __init__(self, dataset: Dataset, using: Transform):
        self.dtst = dataset
        self.using = using

    def __len__(self):
        return len(self.dtst)

    def __getitem__(self, index):
        image, label = self.dtst[index]
        if self.using:
            image = self.using(image)
        return image, label


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
        assert os.path.isdir(
            images_folder_abs_path
        ), "The `images_folder_abs_path` provided is neither valid nor it's pointing to a folder."
        assert os.path.exists(
            images_labels_csv_abs_path
        ), "The `images_labels_csv_abs_path` provided isn't valid"
        assert (
            sum(train_val_test) == 1.0
        ), "The percentages of the train/val/test partitions must add up to 1."
        assert batch_size >= 1, "The batch size should be at least 1."

        self.imgs_path = images_folder_abs_path
        self.imgs_csv = images_labels_csv_abs_path
        self.train_val_test = train_val_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imgs_path = images_folder_abs_path
        self.imgs_df = pd.read_csv(
            images_labels_csv_abs_path, names=["rel_path", "lbl"]
        )

    def setup(self, stage):
        def is_valid_image(row, images_folder_abs_path):
            rel_path = row["rel_path"]
            full_path = os.path.join(images_folder_abs_path, rel_path)
            try:
                with Image.open(full_path) as img:
                    img.verify()  # verifica che l'immagine non sia corrotta
                return row
            except Exception as e:
                return None

        def filter_valid_images(imgs_df, images_folder_abs_path, max_workers=8):
            valid_entries = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(is_valid_image, row, images_folder_abs_path)
                    for _, row in imgs_df.iterrows()
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        valid_entries.append(result)

            return pd.DataFrame(valid_entries)

        self.imgs_df = filter_valid_images(
            self.imgs_df, self.imgs_path, os.cpu_count() - 1
        )

        dtst = BIDataset(
            self.imgs_df,
            transform=t.Compose(
                [
                    t.ToDtype(torch.float32, scale=False),
                    t.Resize(300),
                ]
            ),
        )
        train_set_augmentations = t.Compose(
            t.RandomHorizontalFlip(0.5),
            t.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            t.RandomPerspective(distortion_scale=0.25, p=0.5),
            t.ToDtype(torch.float32, scale=True),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        evaluation_set_agumentations = t.Compose(
            t.ToDtype(torch.float32, scale=True),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        train_set, val_set, test_set = random_split(dtst, self.train_val_test)

        if stage in ("fit", "validate", None):
            self.val_set = Augment(dataset=val_set, using=evaluation_set_agumentations)
        if stage in ("fit", None):
            self.train_set = Augment(dataset=train_set, using=train_set_augmentations)
        if stage in ("test", None):
            self.test_set = Augment(test_set, using=evaluation_set_agumentations)

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
