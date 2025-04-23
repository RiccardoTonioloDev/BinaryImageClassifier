from BinaryImageClassifier import BIClassifier, FitDataManager, Config
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger

import os


def main():
    seed_everything(42, workers=True)
    conf = Config()

    wandb_logger = WandbLogger(project="BIC_Etra", name=conf.exp_name, log_model=True)

    checkpoint_acc = ModelCheckpoint(
        monitor="val/epoch_acc",
        mode="max",
        save_top_k=1,
        filename="best-acc-{epoch:02d}-{val_epoch_acc:.4f}",
        dirpath=os.path.join(conf.checkpoint_path, conf.exp_name),
    )

    checkpoint_prec = ModelCheckpoint(
        monitor="val/epoch_prec",
        mode="max",
        save_top_k=1,
        filename="best-prec-{epoch:02d}-{val_epoch_prec:.4f}",
        dirpath=os.path.join(conf.checkpoint_path, conf.exp_name),
    )

    checkpoint_rec = ModelCheckpoint(
        monitor="val/epoch_rec",
        mode="max",
        save_top_k=1,
        filename="best-rec-{epoch:02d}-{val_epoch_rec:.4f}",
        dirpath=os.path.join(conf.checkpoint_path, conf.exp_name),
    )

    trainer = Trainer(
        max_epochs=conf.max_epochs,
        accelerator=conf.accelerator,
        devices=conf.devices,
        precision=conf.precision,
        num_sanity_val_steps=conf.num_sanity_val_steps,
        callbacks=[checkpoint_acc, checkpoint_rec, checkpoint_prec],
        fast_dev_run=conf.fast_dev_run,
        inference_mode=conf.inference_mode,
    )

    model = BIClassifier(conf.lr)

    data = FitDataManager(
        conf.images_folder_abs_path,
        conf.images_labels_csv_abs_path,
        batch_size=conf.batch_size,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
