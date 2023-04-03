from argparse import ArgumentParser

from typing import Any, Optional, Union
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from thermaldata import Thermal, collate_fn
# from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from torchvision.transforms import Grayscale, Resize
# from pytorch_metric_learning import losses
import wandb
from torch import nn
from datetime import datetime
import importlib
from kaist import Kaist
import wandb

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class CustomSaver(Callback):
    def __init__(self, n_epochs, dir, wandb_dir=None):
        self.total_count = 0
        self.n_epochs = n_epochs
        self.dir = dir
        self.wandb_dir = wandb_dir
        self.counter = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.counter += 1
        self.total_count += 1
        if self.counter == self.n_epochs:
            trainer.save_checkpoint(self.dir + '/debug_'+ str(self.total_count) + '.ckpt')
            self.counter = 0

def train():
    torch.cuda.empty_cache()

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H.%M.%S")
    path = './trained_models/'
    model_dict = dict()
    from light_gan import LightFlexCondGAN

    model = LightFlexCondGAN(lambda_adv=0.05, lambda_cycle=1, lambda_l1=1, lambda_edge=1, lambda_ssim=1)
    model_dict['norm_benchmark'] = model
    model = LightFlexCondGAN(lambda_adv=1, lambda_cycle=20, lambda_l1=20, lambda_edge=20,lambda_ssim=20)
    model_dict['unnorm_benchmark'] = model
    model = LightFlexCondGAN(lambda_adv=1, lambda_cycle=10, lambda_l1=10, lambda_edge=10, lambda_ssim=10)
    model_dict['ebit'] = model
    model = LightFlexCondGAN(lambda_adv=0.5, lambda_cycle=10, lambda_l1=10, lambda_edge=10, lambda_ssim=10)
    model_dict['patel'] = model
    model = LightFlexCondGAN(lambda_adv=1, lambda_cycle=10, lambda_l1=0, lambda_edge=1, lambda_ssim=1)
    model_dict['luo_e'] = model
    model = LightFlexCondGAN(lambda_adv=1, lambda_cycle=10, lambda_l1=1, lambda_edge=1, lambda_ssim=10)
    model_dict['explore_f'] = model
    model = LightFlexCondGAN(lambda_adv=0.05, lambda_cycle=1, lambda_l1=0.5, lambda_edge=0.5, lambda_ssim=1)
    model_dict['explore_g'] = model

    seed_everything()

    n_epochs = 400

    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 1,
              'collate_fn': collate_fn,
              'drop_last': True,
              'pin_memory': True}


    print('Using kaist')
    input_path = "../kaist/kaist-cvpr15"
    # ds = Kaist(input_path, set_name="imageSets/train-all-01.txt")
    # val_ds = Kaist(input_path, set_name="imageSets/test-all-20.txt")
    ds = Kaist(input_path, set_name="train_cut")
    val_ds = Kaist(input_path, set_name="test_cut")


    data_loader = torch.utils.data.DataLoader(
        ds, **params)

    val_data_loader = torch.utils.data.DataLoader(
        val_ds, **params)

    #wandb.init(project=description, group=description+date_time)

    val_checkpoint_interval = 50

    for model_alias in model_dict.keys():
        model = model_dict[model_alias]

        save_dir = path + date_time + model_alias
        saver = CustomSaver(val_checkpoint_interval, dir=save_dir, wandb_dir=model_alias)
        # early_stopping = EarlyStopping('Validation SSIM')
        wandb_logger = WandbLogger(project="warmup",
                            name=model_alias,
                            id=model_alias+date_time,
                            log_model="all",
                            resume="never")
        wandb_logger.watch(model)

        print('Training', model_alias)


        trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=True),
                              accelerator="gpu", logger=wandb_logger,
                              gpus=[1,2,3,5,6,7],
                              check_val_every_n_epoch=val_checkpoint_interval,
                              max_epochs=n_epochs,
                              callbacks=[saver])
        trainer.fit(model, train_dataloaders=data_loader, val_dataloaders=val_data_loader)
        wandb.finish()

if __name__ == "__main__":
    train()