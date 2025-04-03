from BinaryImageClassifier import BIClassifier, FitDataManager
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything


def main():
    seed_everything(42, workers=True)
    LightningCLI(BIClassifier, FitDataManager)


if __name__ == "__main__":
    main()
