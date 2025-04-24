from argparse import ArgumentParser
import os


class Config:
    def __init__(self):
        self._parser = ArgumentParser()

        self._parser.add_argument("--exp_name", type=str)
        self._parser.add_argument("--lr", type=float, default=1e-3)
        self._parser.add_argument("--images_folder_abs_path", type=str)
        self._parser.add_argument("--images_labels_csv_abs_path", type=str)
        self._parser.add_argument("--batch_size", type=int)
        self._parser.add_argument("--max_epochs", type=int)
        self._parser.add_argument("--accelerator", type=str, default="auto")
        self._parser.add_argument("--devices", type=int, default=1)
        self._parser.add_argument("--num_sanity_val_steps", type=int, default=2)
        self._parser.add_argument("--precision", type=int, default=32)
        self._parser.add_argument("--inference_mode", action="store_true")
        self._parser.add_argument("--checkpoint_path", type=str)
        self._parser.add_argument("--fast_dev_run", action="store_true")
        self._parser.add_argument("--num_workers", type=int, default=0)

        args = self._parser.parse_args()
        self.exp_name = args.exp_name
        self.lr = args.lr
        self.images_folder_abs_path = args.images_folder_abs_path
        self.images_labels_csv_abs_path = args.images_labels_csv_abs_path
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.accelerator = args.accelerator
        self.devices = args.devices
        self.num_sanity_val_steps = args.num_sanity_val_steps
        self.precision = args.precision
        self.inference_mode = args.inference_mode
        self.checkpoint_path = args.checkpoint_path
        self.fast_dev_run = args.fast_dev_run
        self.num_workers = args.num_workers

        if not os.path.isdir(os.path.join(self.checkpoint_path, self.exp_name)):
            os.mkdir(os.path.join(self.checkpoint_path, self.exp_name))
