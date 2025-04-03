from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics import Accuracy, Precision, Recall
from typing import Literal, Tuple
from torch import Tensor

import torch.nn.functional as F
import torch.nn as nn
import lightning as L
import torch
import wandb


def custom_resnet():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


class BIClassifier(L.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # MODEL
        self.model = custom_resnet()

        # METRICS - Accuracy, Precision and Recall
        self.batch_train_acc = Accuracy(task="binary")
        self.train_acc = Accuracy(task="binary")
        self.eval_acc = Accuracy(task="binary")
        self.batch_train_prec = Precision(task="binary")
        self.train_prec = Precision(task="binary")
        self.eval_prec = Precision(task="binary")
        self.batch_train_rec = Recall(task="binary")
        self.train_rec = Recall(task="binary")
        self.eval_rec = Recall(task="binary")

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        input, target = batch
        pred: Tensor = self.model.forward(input)
        loss = F.binary_cross_entropy(pred, target)

        # Calculating metrics
        pred = pred.sigmoid()
        batch_acc = self.batch_train_acc(pred, target)
        self.train_acc.update(pred, target)
        batch_prec = self.batch_train_prec(pred, target)
        self.train_prec.update(pred, target)
        batch_rec = self.batch_train_rec(pred, target)
        self.train_rec.update(pred, target)

        # Logging both to wandb and on the stdout
        dict_to_log = {
            "train/batch_loss": loss.to("cpu").item(),
            "train/batch_acc": batch_acc.to("cpu").item(),
            "train/batch_prec": batch_prec.to("cpu").item(),
            "train/batch_rec": batch_rec.to("cpu").item(),
        }
        wandb.log(dict_to_log)
        self.log_dict(
            dict_to_log,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self):
        training_set_acc = self.train_acc.compute()
        training_set_prec = self.train_prec.compute()
        training_set_rec = self.train_rec.compute()
        self.log_dict(
            {
                "train/epoch_acc": training_set_acc,
                "train/epoch_prec": training_set_prec,
                "train/epoch_rec": training_set_rec,
            }
        )
        self.train_acc.reset()
        self.train_prec.reset()
        self.train_rec.reset()

    def _shared_validation_logic(
        self, batch: Tuple[Tensor, Tensor], prefix: Literal["val", "test"]
    ):
        input, target = batch
        pred: Tensor = self.model.forward(input)
        loss = F.binary_cross_entropy(pred, target)

        # Calculating metrics
        pred = pred.sigmoid()
        self.eval_acc.update(pred, target)
        self.eval_prec.update(pred, target)
        self.eval_rec.update(pred, target)

        self.log(f"{prefix}/loss", loss, prog_bar=True)

    def _shared_evaluation_logic(self, prefix: Literal["val", "test"]):
        set_acc = self.eval_acc.compute()
        set_prec = self.eval_prec.compute()
        set_rec = self.eval_rec.compute()
        self.log_dict(
            {
                f"{prefix}/epoch_acc": set_acc,
                f"{prefix}/epoch_prec": set_prec,
                f"{prefix}/epoch_rec": set_rec,
            }
        )
        self.eval_acc.reset()
        self.eval_prec.reset()
        self.eval_rec.reset()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_validation_logic(batch, "val")

    def on_validation_epoch_end(self):
        self._shared_evaluation_logic(self, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_validation_logic(batch, "test")

    def on_test_epoch_end(self):
        self._shared_evaluation_logic("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
