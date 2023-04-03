from argparse import ArgumentParser

from typing import Any, Optional, Union
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from thermaldata import Thermal, collate_fn
# from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchvision.transforms import Grayscale, Resize
# from pytorch_metric_learning import losses
import wandb
from torch import nn
from datetime import datetime
import importlib
from kaist import Kaist

class CustomSaver(Callback):
    def __init__(self, n_epochs, dir, wandb_dir=None):
        self.n_epochs = n_epochs
        self.dir = dir
        self.wandb_dir = wandb_dir
        self.counter = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.counter += 1
        if self.counter == self.n_epochs:
            trainer.save_checkpoint(self.dir)
            self.counter = 0

def train():
    parser = ArgumentParser(description='Train')
    parser.add_argument('--desc', type=str,
                        help='Training description')
    parser.add_argument('--module', type=str,
                        help='Module of model class')
    parser.add_argument('--class_name', type=str,
                        help='Model Class Name')
    parser.add_argument('--gpus', nargs="+", type=int,
                        help='GPU number')
    parser.add_argument('--restore', type=str,
                        help='Path from file to restore')
    parser.add_argument('--wandb_id', type=str,
                        help='Run id to restore')
    parser.add_argument('--ds', type=str,
                        help='Dataset')

    parser.add_argument('--adv', type=float,
                        help='##')

    parser.add_argument('--cycle', action='store_true',
                        help='##')
    parser.add_argument('--edge', action='store_true',
                        help='##')
    parser.add_argument('--ssim', action='store_true',
                        help='##')

    args = parser.parse_args()

    print('#Loading training dataset')
    th_dir = './sonel/thermal'
    cl_dir = './sonel/optical'
    ann_path = './sonel/sonel.csv'
    dev = args.gpus
    ds_name = args.ds
    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H.%M.%S")
    path = './trained_models/'
    description = args.desc
    mod_name = args.module
    class_name = args.class_name
    module = importlib.import_module(mod_name)
    Model = getattr(module, class_name)

    if args.adv:
        model = Model(lambda_adv=args.adv, cycle=args.cycle, edge=args.edge, ssim=args.ssim)
    else:
        model = Model()

    seed_everything()

    n_epochs = 80000

    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1,
              'collate_fn': collate_fn,
              'drop_last': True,
              'pin_memory': True}

    if ds_name == 'kaist':
        print('Using kaist')
        input_path = "../kaist/kaist-cvpr15"
        ds = Kaist(input_path, set_name="train_cut")
        val_ds = Kaist(input_path, set_name="val_cut")
    else:
        ds = Thermal(thermal_dir=th_dir, color_dir=cl_dir, annot_path=ann_path, split='train', normalize=True,
                     label_filter=[2, 3, 5])

        val_ds = Thermal(thermal_dir=th_dir, color_dir=cl_dir, annot_path=ann_path, split='validate', normalize=True,
                         label_filter=[2, 3, 5])

    data_loader = torch.utils.data.DataLoader(
        ds, **params)

    val_data_loader = torch.utils.data.DataLoader(
        val_ds, **params)

    if args.restore:
        resume_type = "must"
    else:
        resume_type = "allow"



    #wandb.init(project=description, group=description+date_time)
    save_dir = path + date_time + description + '/debug.ckpt'
    saver = CustomSaver(100, dir=save_dir, wandb_dir=description)
    wandb_logger = WandbLogger(project=description,
                        name=description,
                        id=args.wandb_id,
                        log_model="all",
                        resume=resume_type)
    wandb_logger.watch(model)

    print('Training')

    if args.restore:
        trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=False),
                          accelerator="gpu", logger=wandb_logger,
                          gpus=dev,
                          check_val_every_n_epoch=100,
                          max_epochs=n_epochs,
                          callbacks=[saver],
                          resume_from_checkpoint=args.restore)
    else:
        trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=True),
                          accelerator="gpu", logger=wandb_logger,
                          gpus=dev,
                          check_val_every_n_epoch=100,
                          max_epochs=n_epochs,
                          callbacks=[saver])
                          #resume_from_checkpoint='trained_models/test_dual_gan_continued08.15.2022-11.22.55/duaL_gan/version_0/checkpoints/debug.ckpt')
    trainer.fit(model, train_dataloaders=data_loader, val_dataloaders=val_data_loader)


if __name__ == "__main__":
    train()