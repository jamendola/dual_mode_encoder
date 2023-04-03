from argparse import ArgumentParser
from typing import Any, Optional, Union
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from thermaldata import Thermal, collate_fn
# from pl_bolts.models.detection.faster_rcnn import create_fasterrcnn_backbone
# from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchvision.transforms import Grayscale, Resize
# from pytorch_metric_learning import losses
import wandb
from torch import nn
import numpy as np
from vgg_ae_variants import VGGAE
import wandb
# from vgg_thermal_ae import DecoderBlock, DecoderLayer
from datetime import datetime
from vgg_gan import PatchGAN, _weights_init, DownSampleConv
import kornia
from ssim import SSIM
# from pytorch_metric_learning import losses



class SimpleDisc(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn

class LEncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super().__init__()

        if layers == 1:

            layer = LEncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = LEncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = LEncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = LEncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x

class LEncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super().__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)

class LVGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False):
        super().__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = LEncoderBlock(input_dim=1, output_dim=64, hidden_dim=64, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = LEncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = LEncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = LEncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = LEncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class LeakyVGGEncoder(nn.Module):

    def __init__(self, enable_bn=False):
        super().__init__()
        self.conv1 = LEncoderBlock(input_dim=1, output_dim=64, hidden_dim=64, layers=2, enable_bn=enable_bn)
        self.conv2 = LEncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=2, enable_bn=enable_bn)
        self.conv3 = LEncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=3, enable_bn=enable_bn)
        self.conv4 = LEncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=3, enable_bn=enable_bn)
        self.conv5 = LEncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=3, enable_bn=enable_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class LDecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super().__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        #print('Debug layer', x.size())
        return self.layer(x)

class LDecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super().__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = LDecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = LDecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = LDecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = LDecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d DecoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
         #   print('Debug layers', x.size())
            x = layer(x)

        return x

class LightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a = self.conv_a(x)
        x_b = self.conv_b(x)
        x_c = self.conv_c(x)
        x = torch.cat((x_a, x_b, x_c), 1)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        return x



class LightEncoderB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))
        self.dil_conv = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=5, stride=1, padding=16,
                      dilation=15),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a = self.conv_a(x)
        x_b = self.conv_b(x)
        x_c = self.conv_c(x)
        x = torch.cat((x_a, x_b, x_c), 1)
        x = self.max_pool(x)
        att = self.dil_conv(x)
        x = self.conv2(x)
        # print('debug', x.size(), att.size())
        x = torch.mul(x, att)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_det = kornia.filters.Canny()

    def forward(self, x):
        x_edge = self.edge_det(x)
        return x_edge


class AdaptedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LDecoderBlock(input_dim=64, output_dim=512, hidden_dim=64, layers=3, enable_bn=True)
        self.conv2 = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv3 = LDecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=3, enable_bn=True)
        self.conv4 = LDecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=2, enable_bn=True)
        self.conv5 = LDecoderBlock(input_dim=64, output_dim=1, hidden_dim=64, layers=2, enable_bn=True)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x

class LeakyVGGRGBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LDecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=3, enable_bn=True)
        self.conv2 = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv3 = LDecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=3, enable_bn=True)
        self.conv4 = LDecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=2, enable_bn=True)
        self.conv5 = LDecoderBlock(input_dim=64, output_dim=3, hidden_dim=64, layers=2, enable_bn=True)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x


class LeakyVGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LDecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=3, enable_bn=True)
        self.conv2 = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv3 = LDecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=3, enable_bn=True)
        self.conv4 = LDecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=2, enable_bn=True)
        self.conv5 = LDecoderBlock(input_dim=64, output_dim=1, hidden_dim=64, layers=2, enable_bn=True)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x


class LeakyDualDecoder(nn.Module):
    def __init__(self, out_chan=1):
        super().__init__()
        self.conv1a = LDecoderBlock(input_dim=64, output_dim=512, hidden_dim=64, layers=3, enable_bn=True)
        self.conv1b = LDecoderBlock(input_dim=64, output_dim=512, hidden_dim=64, layers=3, enable_bn=True)
        self.conv2a = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv2b = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv3 = LDecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=3, enable_bn=True)
        self.conv4 = LDecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=2, enable_bn=True)
        self.conv5 = LDecoderBlock(input_dim=64, output_dim=out_chan, hidden_dim=64, layers=2, enable_bn=True)
        # self.gate = nn.Sigmoid()

    def forward(self, x, mode):
        if mode == 'thermal':
            x = self.conv1a(x)
            x = self.conv2a(x)
        else:
            x = self.conv1b(x)
            x = self.conv2b(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x

class RGBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LDecoderBlock(input_dim=64, output_dim=512, hidden_dim=64, layers=3, enable_bn=True)
        self.conv2 = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv3 = LDecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=3, enable_bn=True)
        self.conv4 = LDecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=2, enable_bn=True)
        self.conv5 = LDecoderBlock(input_dim=64, output_dim=3, hidden_dim=64, layers=2, enable_bn=True)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x

class AdaptedAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightEncoder()
        self.gray_decoder = AdaptedDecoder()
        self.thermal_decoder = AdaptedDecoder()
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)

    def forward(self, x, mode='thermal'):
        z = self.encoder(x)
        #print(z.size())
        # print('Decoding on mode: ', mode)
        if mode == 'thermal':
            x_hat = self.thermal_decoder(z)
        elif mode == 'grayscale':
            x_hat = self.gray_decoder(z)
        return x_hat

class LeakyVGGAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LeakyVGGEncoder()
        self.g2rgb_decoder = LeakyVGGRGBDecoder()
        self.t2t_decoder = LeakyVGGDecoder()
        self.g2t_decoder = LeakyVGGDecoder()
        self.t2rgb_decoder = LeakyVGGRGBDecoder()
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)

    def forward(self, x, mode):
        z = self.encoder(x)
        #print(z.size())
        # print('Decoding on mode: ', mode)
        if mode == 'g2rgb':
            x_hat = self.g2rgb_decoder(z)
        elif mode == 't2t':
            x_hat = self.t2t_decoder(z)
        if mode == 'g2t':
            x_hat = self.g2t_decoder(z)
        elif mode == 't2rgb':
            x_hat = self.t2rgb_decoder(z)
        return x_hat



class AE4D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightEncoder()
        self.g2rgb_decoder = RGBDecoder()
        self.t2t_decoder = AdaptedDecoder()
        self.g2t_decoder = AdaptedDecoder()
        self.t2rgb_decoder = RGBDecoder()
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)

    def forward(self, x, mode):
        z = self.encoder(x)
        #print(z.size())
        # print('Decoding on mode: ', mode)
        if mode == 'g2rgb':
            x_hat = self.g2rgb_decoder(z)
        elif mode == 't2t':
            x_hat = self.t2t_decoder(z)
        if mode == 'g2t':
            x_hat = self.g2t_decoder(z)
        elif mode == 't2rgb':
            x_hat = self.t2rgb_decoder(z)
        return x_hat

class AE4DB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightEncoderB()
        self.g2rgb_decoder = RGBDecoder()
        self.t2t_decoder = AdaptedDecoder()
        self.g2t_decoder = AdaptedDecoder()
        self.t2rgb_decoder = RGBDecoder()
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)

    def forward(self, x, mode):
        z = self.encoder(x)
        #print(z.size())
        # print('Decoding on mode: ', mode)
        if mode == 'g2rgb':
            x_hat = self.g2rgb_decoder(z)
        elif mode == 't2t':
            x_hat = self.t2t_decoder(z)
        if mode == 'g2t':
            x_hat = self.g2t_decoder(z)
        elif mode == 't2rgb':
            x_hat = self.t2rgb_decoder(z)
        return x_hat

class AE2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightEncoder()
        self.rgb_decoder = RGBDecoder()
        self.thermal_decoder = AdaptedDecoder()
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)

    def forward(self, x, mode='thermal'):
        z = self.encoder(x)
        if mode == 'thermal':
            x_hat = self.thermal_decoder(z)
        elif mode == 'rgb':
            x_hat = self.rgb_decoder(z)
        return x_hat, z


class AE2DDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightEncoder()
        self.rgb_decoder = LeakyDualDecoder(out_chan=3)
        self.thermal_decoder = LeakyDualDecoder(out_chan=1)
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)

    def forward(self, x, mode='thermal'):
        z = self.encoder(x)
        if mode == 'g2rgb':
            x_hat = self.rgb_decoder(z, mode='grayscale')
        elif mode == 't2t':
            x_hat = self.thermal_decoder(z, mode='thermal')
        if mode == 'g2t':
            x_hat = self.thermal_decoder(z, mode='grayscale')
        elif mode == 't2rgb':
            x_hat = self.rgb_decoder(z, mode='thermal')
        return x_hat


class LES4DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = AE4D()
        self.color_disc = PatchGAN(6)
        self.thermal_disc = PatchGAN(2)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal, x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb, x_rgb)
        real_rgb_logits = self.color_disc(x_rgb, x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb, x_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        # loss = None
        loss_discr_gray = self._disc_step_rgb(x_gray, x_thermal, x_color)
        self.log("Validation RGB Discriminator Loss", loss_discr_gray, sync_dist=True)
        loss_discr_thermal = self._disc_step_thermal(x_gray, x_thermal)
        self.log("Validation Thermal Discriminator Loss", loss_discr_thermal, sync_dist=True)
        # loss = None
        return loss_gen+loss_discr_gray+loss_discr_thermal


class LES4G4DCondGAN(LightningModule): #TODO
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = AE4D()
        self.g2rgb_disc = PatchGAN(4)
        self.t2t_disc = PatchGAN(2)
        self.t2rgb_disc = PatchGAN(4)
        self.g2t_disc = PatchGAN(2)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.g2rgb_disc = self.g2rgb_disc.apply(_weights_init)
        self.t2t_disc = self.t2t_disc.apply(_weights_init)
        self.t2rgb_disc = self.t2rgb_disc.apply(_weights_init)
        self.g2t_disc = self.g2t_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.t2t_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.g2rgb_disc(rec_rgb, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.g2t_disc(trans_thermal, x_gray)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.t2rgb_disc(trans_rgb, x_thermal)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_t2t(self, x_thermal):
        rec_t2t = self.gen(x_thermal, mode='t2t').detach()
        rec_t2t_logits = self.t2t_disc(rec_t2t, x_thermal)
        real_t2t_logits = self.t2t_disc(x_thermal, x_thermal)

        fake_t2t_loss = self.adversarial_criterion(rec_t2t_logits, torch.zeros_like(rec_t2t_logits))
        real_t2t_loss = self.adversarial_criterion(real_t2t_logits, torch.ones_like(real_t2t_logits))

        loss = fake_t2t_loss + real_t2t_loss
        return loss

    def _disc_step_g2t(self, x_gray, x_thermal):
        rec_g2t = self.gen(x_gray, mode='g2t').detach()
        rec_g2t_logits = self.g2t_disc(rec_g2t, x_gray)
        real_g2t_logits = self.g2t_disc(x_thermal, x_gray)

        fake_g2t_loss = self.adversarial_criterion(rec_g2t_logits, torch.zeros_like(rec_g2t_logits))
        real_g2t_loss = self.adversarial_criterion(real_g2t_logits, torch.ones_like(real_g2t_logits))

        loss = fake_g2t_loss + real_g2t_loss
        return loss

    def _disc_step_g2rgb(self, x_gray, x_rgb):
        rec_g2rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_g2rgb_logits = self.g2rgb_disc(rec_g2rgb, x_gray)
        real_g2rgb_logits = self.g2rgb_disc(x_rgb, x_gray)

        fake_g2rgb_loss = self.adversarial_criterion(rec_g2rgb_logits, torch.zeros_like(rec_g2rgb_logits))
        real_g2rgb_loss = self.adversarial_criterion(real_g2rgb_logits, torch.ones_like(real_g2rgb_logits))

        loss = fake_g2rgb_loss + real_g2rgb_loss
        return loss

    def _disc_step_t2rgb(self, x_thermal, x_rgb):
        rec_t2rgb = self.gen(x_thermal, mode='t2rgb').detach()
        rec_t2rgb_logits = self.t2rgb_disc(rec_t2rgb, x_thermal)
        real_t2rgb_logits = self.t2rgb_disc(x_rgb, x_thermal)

        fake_t2rgb_loss = self.adversarial_criterion(rec_t2rgb_logits, torch.zeros_like(rec_t2rgb_logits))
        real_t2rgb_loss = self.adversarial_criterion(real_t2rgb_logits, torch.ones_like(real_t2rgb_logits))

        loss = fake_t2rgb_loss + real_t2rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_g2rgb_opt = torch.optim.Adam(self.g2rgb_disc.parameters(), lr=lr)
        disc_t2t_opt = torch.optim.Adam(self.t2t_disc.parameters(), lr=lr)
        disc_g2t_opt = torch.optim.Adam(self.g2t_disc.parameters(), lr=lr)
        disc_t2rgb_opt = torch.optim.Adam(self.t2rgb_disc.parameters(), lr=lr)
        return gen_opt, disc_g2rgb_opt, disc_t2t_opt, disc_g2t_opt, disc_t2rgb_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0: #gen
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:#g2rgb
            loss = self._disc_step_g2rgb(x_gray, x_color)
            self.log("Train Gray to RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:#t2t
            loss = self._disc_step_t2t(x_thermal)
            self.log("Train Thermal to Thermal Discriminator Loss", loss, sync_dist=True)
        elif optimizer_idx == 3:#g2t
            loss = self._disc_step_g2t(x_gray, x_thermal)
            self.log("Train Gray to Thermal Discriminator Loss", loss, sync_dist=True)
        elif optimizer_idx == 4:#t2rgb
            loss = self._disc_step_t2rgb(x_thermal, x_color)
            self.log("Train Thermal to RGB Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)

        return loss_gen

class LES4G2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = AE4D()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

class VGGES4G2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = LeakyVGGAE()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

class CycleLES4G2DEncRegCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE4D()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb


        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb = self.gen(trans_thermal, mode='t2rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal = self.gen(self.gen.gray_transformer(trans_rgb), mode='g2t')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

class CLE2SR4G2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE4D()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        _, rec_thermal_edge = self.edge_det(rec_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        _, rec_rgb_edge = self.edge_det(rec_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb


        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        _, trans_thermal_edge = self.edge_det(trans_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        _, trans_rgb_edge = self.edge_det(trans_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb = self.gen(trans_thermal, mode='t2rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal = self.gen(self.gen.gray_transformer(trans_rgb), mode='g2t')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

class CycleLES4G2DEncBRegCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE4DB()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb


        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb = self.gen(trans_thermal, mode='t2rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal = self.gen(self.gen.gray_transformer(trans_rgb), mode='g2t')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

class CycleLES4DCondGAN(LES4DCondGAN):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = AE4D()
        self.color_disc = PatchGAN(6)
        self.thermal_disc = PatchGAN(2)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb


        #cycle
        cycle_rgb = self.gen(self.gen.gray_transformer(rec_rgb), mode='g2rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal = self.gen(rec_thermal, mode='t2t')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)


        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb = self.gen(trans_thermal, mode='t2rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal = self.gen(self.gen.gray_transformer(trans_rgb), mode='g2t')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

class LS4DCondGAN(LES4DCondGAN):
    def __init__(self):
        super().__init__()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)


        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss  + dsim_rgb)

        return thermal_loss + rgb_loss


class ContrastCLE2SR2G2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE2D()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal, z_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb, z_gray = self.gen(x_gray, mode='rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        _, rec_thermal_edge = self.edge_det(rec_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        _, rec_rgb_edge = self.edge_det(rec_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # #Contrastive
        z_gray_f = torch.flatten(z_gray, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        neg_target = torch.full((z_thermal.size(0),), -1).to(z_thermal.device)
        contrast = 5*(self.cos_loss(z_gray_f, z_thermal_f, neg_target))

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss + contrast

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal,_= self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb,_ = self.gen(x_thermal, mode='rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        _, trans_thermal_edge = self.edge_det(trans_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        _, trans_rgb_edge = self.edge_det(trans_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb, _ = self.gen(trans_thermal, mode='rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal, _ = self.gen(self.gen.gray_transformer(trans_rgb), mode='thermal')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal , _ = self.gen(x_thermal, mode='thermal')
        rec_thermal = rec_thermal.detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal , _ = self.gen(x_gray, mode='thermal')
        trans_thermal = trans_thermal.detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb, _ = self.gen(x_gray, mode='rgb')
        rec_rgb = rec_rgb.detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb, _ = self.gen(x_thermal, mode='rgb')
        trans_rgb = trans_rgb.detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

class TGContrastCLE2SR2G2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE2D()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.constrast_loss = losses.NTXentLoss(temperature=0.07)

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal, z_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb, z_gray = self.gen(x_gray, mode='rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        _, rec_thermal_edge = self.edge_det(rec_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        _, rec_rgb_edge = self.edge_det(rec_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # #Contrastive
        z_gray_f = torch.flatten(z_gray, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_gray_f.size(0)
        gray_labels = torch.full((batch_size,), 0)
        thermal_labels = torch.full((batch_size,), 1)
        labels = torch.cat([gray_labels, thermal_labels], dim=0)
        # print('debug labels', labels.size())
        embeddings = torch.cat([z_gray_f, z_thermal_f], dim=0)
        # print('debug contrast', embeddings.size())
        contrast = self.constrast_loss(embeddings, labels)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss + contrast

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal,_= self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb,_ = self.gen(x_thermal, mode='rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        _, trans_thermal_edge = self.edge_det(trans_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        _, trans_rgb_edge = self.edge_det(trans_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb, _ = self.gen(trans_thermal, mode='rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal, _ = self.gen(self.gen.gray_transformer(trans_rgb), mode='thermal')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal , _ = self.gen(x_thermal, mode='thermal')
        rec_thermal = rec_thermal.detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal , _ = self.gen(x_gray, mode='thermal')
        trans_thermal = trans_thermal.detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb, _ = self.gen(x_gray, mode='rgb')
        rec_rgb = rec_rgb.detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb, _ = self.gen(x_thermal, mode='rgb')
        trans_rgb = trans_rgb.detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen


class CLE2SR2dualG2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE2DDual()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        _, rec_thermal_edge = self.edge_det(rec_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        _, rec_rgb_edge = self.edge_det(rec_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb


        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        _, trans_thermal_edge = self.edge_det(trans_thermal)
        _, x_thermal_edge = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        _, trans_rgb_edge = self.edge_det(trans_rgb)
        _, x_rgb_edge = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_rgb = self.gen(trans_thermal, mode='t2rgb')
        cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb)

        cycle_thermal = self.gen(self.gen.gray_transformer(trans_rgb), mode='g2t')
        cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen


class LES2DCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = AE2D()
        self.color_disc = PatchGAN(6)
        self.thermal_disc = PatchGAN(2)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_rgb = self.gen(x_gray, mode='rgb')
        rec_rgb_logits = self.color_disc(rec_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_rgb_edge, _ = self.edge_det(rec_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (rec_rgb_loss + rgb_edge_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb):
        trans_thermal = self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_rgb = self.gen(x_thermal, mode='rgb')
        trans_rgb_logits = self.color_disc(trans_rgb, x_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_rgb_edge, _ = self.edge_det(trans_rgb)
        x_rgb_edge, _ = self.edge_det(x_rgb)
        gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge)

        ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
        dsim_rgb = 1 - ssim_rgb

        thermal_loss = thermal_adversarial_loss + lambda_recon * (trans_thermal_loss + thermal_edge_loss + dsim_thermal)
        rgb_loss = gray_adversarial_loss + lambda_recon * (trans_rgb_loss + gray_rgb_loss + dsim_rgb)

        return thermal_loss + rgb_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='thermal').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal, x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='thermal').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb, x_rgb)
        real_rgb_logits = self.color_disc(x_rgb, x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits))
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits))

        trans_rgb = self.gen(x_thermal, mode='rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb, x_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits))

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        # loss = None
        loss_discr_gray = self._disc_step_rgb(x_gray, x_thermal, x_color)
        self.log("Validation RGB Discriminator Loss", loss_discr_gray, sync_dist=True)
        loss_discr_thermal = self._disc_step_thermal(x_gray, x_thermal)
        self.log("Validation Thermal Discriminator Loss", loss_discr_thermal, sync_dist=True)
        # loss = None
        return loss_gen+loss_discr_gray+loss_discr_thermal

class LightCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 200
        self.learning_rate = 0.0002
        self.gen = AdaptedAE()
        self.patch_gan = PatchGAN(2)

        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.patch_gan(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale')
        rec_gray_logits = self.patch_gan(rec_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(rec_gray_logits, torch.ones_like(rec_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(rec_gray, x_gray)

        thermal_loss = thermal_adversarial_loss + lambda_recon * rec_thermal_loss
        gray_loss = gray_adversarial_loss + lambda_recon * rec_gray_loss

        return thermal_loss + gray_loss

    def _gen_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.patch_gan(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale')
        trans_gray_logits = self.patch_gan(trans_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(trans_gray_logits, torch.ones_like(trans_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(trans_gray, x_gray)

        thermal_loss = thermal_adversarial_loss + lambda_recon * rec_thermal_loss
        gray_loss = gray_adversarial_loss + lambda_recon * rec_gray_loss

        return thermal_loss + gray_loss

    def _disc_step_direct(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='thermal').detach()
        rec_thermal_logits = self.patch_gan(rec_thermal, x_thermal)
        real_thermal_logits = self.patch_gan(x_thermal, x_thermal)


        fake_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale').detach()
        rec_gray_logits = self.patch_gan(rec_gray, x_gray)
        real_gray_logits = self.patch_gan(x_gray, x_gray)

        fake_gray_loss = self.adversarial_criterion(rec_gray_logits, torch.zeros_like(rec_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        loss = (fake_thermal_loss + real_thermal_loss) / 2
        loss = loss + (fake_gray_loss + real_gray_loss) / 2
        return loss

    def _disc_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal').detach()
        trans_thermal_logits = self.patch_gan(trans_thermal, x_thermal)
        real_thermal_logits = self.patch_gan(x_thermal, x_thermal)

        trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale').detach()
        trans_gray_logits = self.patch_gan(trans_gray, x_gray)
        real_gray_logits = self.patch_gan(x_gray, x_gray)

        trans_gray_loss = self.adversarial_criterion(trans_gray_logits, torch.zeros_like(trans_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        loss = (trans_thermal_loss + real_thermal_loss) / 2
        loss = loss + (trans_gray_loss + real_gray_loss) / 2
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr)
        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_discr_direct = self._disc_step_direct(x_gray, x_thermal)
            loss_discr_translate = self._disc_step_translate(x_gray, x_thermal)
            loss = loss_discr_direct + loss_discr_translate
            self.log("train_Discriminator Loss", loss)
        elif optimizer_idx == 1:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
            loss = loss_gen_direct + loss_gen_translate
            self.log("train_Generator Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        # loss = None
        loss_discr_direct = self._disc_step_direct(x_gray, x_thermal)
        loss_discr_translate = self._disc_step_translate(x_gray, x_thermal)
        loss_discr = loss_discr_direct + loss_discr_translate
        self.log("val_Discriminator Loss", loss_discr)
        # loss = None
        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("val_Generator Loss", loss_gen)
        return loss_discr+loss_gen


class WeightedCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 200
        self.learning_rate = 0.0002
        self.gen = AdaptedAE()
        self.patch_gan = PatchGAN(2)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.patch_gan(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale')
        rec_gray_logits = self.patch_gan(rec_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(rec_gray_logits, torch.ones_like(rec_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(rec_gray, x_gray)
        # extra incentive for thermal
        thermal_loss = thermal_adversarial_loss + 10 * lambda_recon * rec_thermal_loss
        gray_loss = gray_adversarial_loss + lambda_recon * rec_gray_loss

        return thermal_loss + gray_loss

    def _gen_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.patch_gan(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits,
                                                              torch.ones_like(trans_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale')
        trans_gray_logits = self.patch_gan(trans_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(trans_gray_logits, torch.ones_like(trans_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(trans_gray, x_gray)

        thermal_loss = thermal_adversarial_loss + 10 * lambda_recon * rec_thermal_loss
        gray_loss = gray_adversarial_loss + lambda_recon * rec_gray_loss

        return thermal_loss + gray_loss

    def _disc_step_direct(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='thermal').detach()
        rec_thermal_logits = self.patch_gan(rec_thermal, x_thermal)
        real_thermal_logits = self.patch_gan(x_thermal, x_thermal)

        fake_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale').detach()
        rec_gray_logits = self.patch_gan(rec_gray, x_gray)
        real_gray_logits = self.patch_gan(x_gray, x_gray)

        fake_gray_loss = self.adversarial_criterion(rec_gray_logits, torch.zeros_like(rec_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        loss = (fake_thermal_loss + real_thermal_loss) / 2
        loss = loss + (fake_gray_loss + real_gray_loss) / 2
        return loss

    def _disc_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal').detach()
        trans_thermal_logits = self.patch_gan(trans_thermal, x_thermal)
        real_thermal_logits = self.patch_gan(x_thermal, x_thermal)

        trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale').detach()
        trans_gray_logits = self.patch_gan(trans_gray, x_gray)
        real_gray_logits = self.patch_gan(x_gray, x_gray)

        trans_gray_loss = self.adversarial_criterion(trans_gray_logits, torch.zeros_like(trans_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        loss = (trans_thermal_loss + real_thermal_loss) / 2
        loss = loss + (trans_gray_loss + real_gray_loss) / 2
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr)
        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_discr_direct = self._disc_step_direct(x_gray, x_thermal)
            loss_discr_translate = self._disc_step_translate(x_gray, x_thermal)
            loss = loss_discr_direct + loss_discr_translate
            self.log("train_Discriminator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
            loss = loss_gen_direct + loss_gen_translate
            self.log("train_Generator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        # loss = None
        loss_discr_direct = self._disc_step_direct(x_gray, x_thermal)
        loss_discr_translate = self._disc_step_translate(x_gray, x_thermal)
        loss_discr = loss_discr_direct + loss_discr_translate
        self.log("val_Discriminator Loss", loss_discr, sync_dist=True)
        # loss = None
        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("val_Generator Loss", loss_gen, sync_dist=True)
        return loss_discr + loss_gen

class LightEdgeCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 200
        self.learning_rate = 0.0002
        self.gen = AdaptedAE()
        self.patch_gan = PatchGAN(2)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.patch_gan(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale')
        rec_gray_logits = self.patch_gan(rec_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(rec_gray_logits, torch.ones_like(rec_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(rec_gray, x_gray)
        #extra incentive for thermal
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)
        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss)
        gray_loss = gray_adversarial_loss + lambda_recon * rec_gray_loss

        return thermal_loss + gray_loss

    def _gen_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.patch_gan(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale')
        trans_gray_logits = self.patch_gan(trans_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(trans_gray_logits, torch.ones_like(trans_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(trans_gray, x_gray)

        rec_gray_edge, _ = self.edge_det(trans_gray)
        x_gray_edge, _ = self.edge_det(x_gray)
        gray_edge_loss = self.recon_criterion(rec_gray_edge, x_gray_edge)

        thermal_loss = thermal_adversarial_loss + lambda_recon * rec_thermal_loss
        gray_loss = gray_adversarial_loss + lambda_recon * (rec_gray_loss + gray_edge_loss)

        return thermal_loss + gray_loss

    def _disc_step_direct(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='thermal').detach()
        rec_thermal_logits = self.patch_gan(rec_thermal, x_thermal)
        real_thermal_logits = self.patch_gan(x_thermal, x_thermal)


        fake_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale').detach()
        rec_gray_logits = self.patch_gan(rec_gray, x_gray)
        real_gray_logits = self.patch_gan(x_gray, x_gray)

        fake_gray_loss = self.adversarial_criterion(rec_gray_logits, torch.zeros_like(rec_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        loss = (fake_thermal_loss + real_thermal_loss) / 2
        loss = loss + (fake_gray_loss + real_gray_loss) / 2
        return loss

    def _disc_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal').detach()
        trans_thermal_logits = self.patch_gan(trans_thermal, x_thermal)
        real_thermal_logits = self.patch_gan(x_thermal, x_thermal)

        trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale').detach()
        trans_gray_logits = self.patch_gan(trans_gray, x_gray)
        real_gray_logits = self.patch_gan(x_gray, x_gray)

        trans_gray_loss = self.adversarial_criterion(trans_gray_logits, torch.zeros_like(trans_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        loss = (trans_thermal_loss + real_thermal_loss) / 2
        loss = loss + (trans_gray_loss + real_gray_loss) / 2
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr)
        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_discr_direct = self._disc_step_direct(x_gray, x_thermal)
            loss_discr_translate = self._disc_step_translate(x_gray, x_thermal)
            loss = loss_discr_direct + loss_discr_translate
            self.log("train_Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 1:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
            loss = loss_gen_direct + loss_gen_translate
            self.log("train_Generator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        # loss = None
        loss_discr_direct = self._disc_step_direct(x_gray, x_thermal)
        loss_discr_translate = self._disc_step_translate(x_gray, x_thermal)
        loss_discr = loss_discr_direct + loss_discr_translate
        self.log("val_Discriminator Loss", loss_discr,  sync_dist=True)
        # loss = None
        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("val_Generator Loss", loss_gen,  sync_dist=True)
        return loss_discr+loss_gen

class LEDDCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 200
        self.learning_rate = 0.0002
        self.gen = AdaptedAE()
        self.gray_disc = PatchGAN(2)
        self.thermal_disc = PatchGAN(2)

        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.gray_disc = self.gray_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale')
        rec_gray_logits = self.gray_disc(rec_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(rec_gray_logits, torch.ones_like(rec_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(rec_gray, x_gray)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_gray_edge, _ = self.edge_det(rec_gray)
        x_gray_edge, _ = self.edge_det(x_gray)
        gray_edge_loss = self.recon_criterion(rec_gray_edge, x_gray_edge)


        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss)
        gray_loss = gray_adversarial_loss + lambda_recon * (rec_gray_loss + gray_edge_loss)

        return thermal_loss + gray_loss

    def _gen_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale')
        trans_gray_logits = self.gray_disc(trans_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(trans_gray_logits, torch.ones_like(trans_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(trans_gray, x_gray)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_gray_edge, _ = self.edge_det(trans_gray)
        x_gray_edge, _ = self.edge_det(x_gray)
        gray_edge_loss = self.recon_criterion(trans_gray_edge, x_gray_edge)

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss)
        gray_loss = gray_adversarial_loss + lambda_recon * (rec_gray_loss + gray_edge_loss)

        return thermal_loss + gray_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='thermal').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal, x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='thermal').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_gray(self, x_gray, x_thermal):
        rec_gray = self.gen(x_gray, mode='grayscale').detach()
        rec_gray_logits = self.gray_disc(rec_gray, x_gray)
        real_gray_logits = self.gray_disc(x_gray, x_gray)

        fake_rec_gray_loss = self.adversarial_criterion(rec_gray_logits, torch.zeros_like(rec_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale').detach()
        trans_gray_logits = self.gray_disc(trans_gray, x_gray)

        fake_trans_gray_loss = self.adversarial_criterion(trans_gray_logits, torch.zeros_like(trans_gray_logits))

        loss = fake_rec_gray_loss + fake_trans_gray_loss + 2*real_gray_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.gray_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_gray(x_gray, x_thermal)
            self.log("Train Gray Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        # loss = None
        loss_discr_gray = self._disc_step_gray(x_gray, x_thermal)
        self.log("Validation Gray Discriminator Loss", loss_discr_gray, sync_dist=True)
        loss_discr_thermal = self._disc_step_thermal(x_gray, x_thermal)
        self.log("Validation Thermal Discriminator Loss", loss_discr_thermal, sync_dist=True)
        # loss = None
        return loss_gen+loss_discr_gray+loss_discr_thermal


class LESDDCondGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_recon = 20
        self.learning_rate = 0.0002
        self.gen = AdaptedAE()
        self.gray_disc = PatchGAN(2)
        self.thermal_disc = PatchGAN(2)
        self.ssim = SSIM()
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.gray_disc = self.gray_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='thermal')
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits))

        rec_gray = self.gen(x_gray, mode='grayscale')
        rec_gray_logits = self.gray_disc(rec_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(rec_gray_logits, torch.ones_like(rec_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(rec_gray, x_gray)
        #extra incentive
        rec_thermal_edge, _ = self.edge_det(rec_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge)

        rec_gray_edge, _ = self.edge_det(rec_gray)
        x_gray_edge, _ = self.edge_det(x_gray)
        gray_edge_loss = self.recon_criterion(rec_gray_edge, x_gray_edge)

        ssim_thermal = self.ssim(x_thermal, rec_thermal).detach()
        dsim_thermal = 1-ssim_thermal

        ssim_gray = self.ssim(x_gray, rec_gray).detach()
        dsim_gray = 1 - ssim_gray

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        gray_loss = gray_adversarial_loss + lambda_recon * (rec_gray_loss + gray_edge_loss + dsim_gray)

        return thermal_loss + gray_loss

    def _gen_step_translate(self, x_gray, x_thermal):
        trans_thermal = self.gen(x_gray, mode='thermal')
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale')
        trans_gray_logits = self.gray_disc(trans_gray, x_gray)
        gray_adversarial_loss = self.adversarial_criterion(trans_gray_logits, torch.ones_like(trans_gray_logits))

        lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(trans_thermal, x_thermal)
        rec_gray_loss = self.recon_criterion(trans_gray, x_gray)

        trans_thermal_edge, _ = self.edge_det(trans_thermal)
        x_thermal_edge, _ = self.edge_det(x_thermal)
        thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge)

        trans_gray_edge, _ = self.edge_det(trans_gray)
        x_gray_edge, _ = self.edge_det(x_gray)
        gray_edge_loss = self.recon_criterion(trans_gray_edge, x_gray_edge)

        ssim_thermal = self.ssim(x_thermal, trans_thermal).detach()
        dsim_thermal = 1 - ssim_thermal

        ssim_gray = self.ssim(x_gray, trans_gray).detach()
        dsim_gray = 1 - ssim_gray

        thermal_loss = thermal_adversarial_loss + lambda_recon * (rec_thermal_loss + thermal_edge_loss + dsim_thermal)
        gray_loss = gray_adversarial_loss + lambda_recon * (rec_gray_loss + gray_edge_loss + dsim_gray)

        return thermal_loss + gray_loss

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='thermal').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal, x_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal, x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits))
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits))

        trans_thermal = self.gen(x_gray, mode='thermal').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal, x_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_gray(self, x_gray, x_thermal):
        rec_gray = self.gen(x_gray, mode='grayscale').detach()
        rec_gray_logits = self.gray_disc(rec_gray, x_gray)
        real_gray_logits = self.gray_disc(x_gray, x_gray)

        fake_rec_gray_loss = self.adversarial_criterion(rec_gray_logits, torch.zeros_like(rec_gray_logits))
        real_gray_loss = self.adversarial_criterion(real_gray_logits, torch.ones_like(real_gray_logits))

        trans_gray = self.gen(x_thermal, mode='grayscale').detach()
        trans_gray_logits = self.gray_disc(trans_gray, x_gray)

        fake_trans_gray_loss = self.adversarial_criterion(trans_gray_logits, torch.zeros_like(trans_gray_logits))

        loss = fake_rec_gray_loss + fake_trans_gray_loss + 2*real_gray_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.gray_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)
        loss = None
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
            loss = loss_gen_direct + loss_gen_translate
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_gray(x_gray, x_thermal)
            self.log("Train Gray Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_gray = self.gen.gray_transformer(x_color)
        x_gray = self.gen.resize(x_gray)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct = self._gen_step_direct(x_gray, x_thermal)
        loss_gen_translate = self._gen_step_translate(x_gray, x_thermal)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        # loss = None
        loss_discr_gray = self._disc_step_gray(x_gray, x_thermal)
        self.log("Validation Gray Discriminator Loss", loss_discr_gray, sync_dist=True)
        loss_discr_thermal = self._disc_step_thermal(x_gray, x_thermal)
        self.log("Validation Thermal Discriminator Loss", loss_discr_thermal, sync_dist=True)
        # loss = None
        return loss_gen+loss_discr_gray+loss_discr_thermal

class LightFlexCondGAN(LightningModule):
    def __init__(self, lambda_adv=0.05, cycle=True, edge=True, ssim=True, lambda_cycle=1, lambda_l1=1, lambda_edge=1,
                 lambda_ssim=1):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_adv = lambda_adv
        self.lambda_cycle = lambda_cycle
        self.lambda_l1 = lambda_l1
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim
        self.cycle = cycle
        self.edge = edge
        self.ssim = ssim
        print(self.lambda_adv, self.cycle, self.edge, self.ssim)
        self.learning_rate = 0.0002
        self.l2_lambda = 0.01
        self.gen = AE4DB()
        self.color_disc = SimpleDisc(3)
        self.thermal_disc = SimpleDisc(1)
        self.thermal_ssim = SSIM(1)
        self.rgb_ssim = SSIM(3)
        self.edge_det = kornia.filters.Canny()
        # intializing weights
        # self.gen = self.gen.apply(_weights_init)
        self.color_disc = self.color_disc.apply(_weights_init)
        self.thermal_disc = self.thermal_disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step_direct(self, x_gray, x_thermal, x_rgb, val=False):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        rec_thermal = self.gen(x_thermal, mode='t2t')
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(rec_thermal_logits, torch.ones_like(rec_thermal_logits)).detach()

        rec_rgb = self.gen(x_gray, mode='g2rgb')
        rec_rgb_logits = self.color_disc(rec_rgb)
        gray_adversarial_loss = self.adversarial_criterion(rec_rgb_logits, torch.ones_like(rec_rgb_logits)).detach()

        # lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        rec_thermal_loss = self.recon_criterion(rec_thermal, x_thermal).detach()
        rec_rgb_loss = self.recon_criterion(rec_rgb, x_rgb).detach()
        #extra incentive

        thermal_edge_loss = 0
        rgb_edge_loss = 0
        if self.edge:
            rec_thermal_edge, _ = self.edge_det(rec_thermal)
            x_thermal_edge, _ = self.edge_det(x_thermal)
            thermal_edge_loss = self.recon_criterion(rec_thermal_edge, x_thermal_edge).detach()

            rec_rgb_edge, _ = self.edge_det(rec_rgb)
            x_rgb_edge, _ = self.edge_det(x_rgb)
            rgb_edge_loss = self.recon_criterion(rec_rgb_edge, x_rgb_edge).detach()

        dsim_thermal = 0
        dsim_rgb = 0
        if self.ssim:
            ssim_thermal = self.thermal_ssim(x_thermal, rec_thermal).detach()
            dsim_thermal = 1-ssim_thermal

            ssim_rgb = self.rgb_ssim(x_rgb, rec_rgb).detach()
            dsim_rgb = 1 - ssim_rgb

        thermal_loss = self.lambda_adv*thermal_adversarial_loss + self.lambda_l1*rec_thermal_loss + \
                       self.lambda_edge*thermal_edge_loss + self.lambda_ssim*dsim_thermal
        rgb_loss = self.lambda_adv*gray_adversarial_loss + self.lambda_l1*rec_rgb_loss + \
                   self.lambda_edge*rgb_edge_loss + self.lambda_ssim*dsim_rgb

        if val:
            return (thermal_loss + rgb_loss), (ssim_thermal + ssim_rgb) / 2
        else:
            return (thermal_loss + rgb_loss)

    def _gen_step_translate(self, x_gray, x_thermal, x_rgb, val=False):
        trans_thermal = self.gen(x_gray, mode='g2t')
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        thermal_adversarial_loss = self.adversarial_criterion(trans_thermal_logits, torch.ones_like(trans_thermal_logits)).detach()

        trans_rgb = self.gen(x_thermal, mode='t2rgb')
        trans_rgb_logits = self.color_disc(trans_rgb)
        gray_adversarial_loss = self.adversarial_criterion(trans_rgb_logits, torch.ones_like(trans_rgb_logits)).detach()

        # lambda_recon = self.lambda_recon
        # calculate reconstruction loss
        trans_thermal_loss = self.recon_criterion(trans_thermal, x_thermal).detach()
        trans_rgb_loss = self.recon_criterion(trans_rgb, x_rgb).detach()

        thermal_edge_loss = 0
        gray_rgb_loss = 0
        if self.edge:

            trans_thermal_edge, _ = self.edge_det(trans_thermal)
            x_thermal_edge, _ = self.edge_det(x_thermal)
            thermal_edge_loss = self.recon_criterion(trans_thermal_edge, x_thermal_edge).detach()

            trans_rgb_edge, _ = self.edge_det(trans_rgb)
            x_rgb_edge, _ = self.edge_det(x_rgb)
            gray_rgb_loss = self.recon_criterion(trans_rgb_edge, x_rgb_edge).detach()

        dsim_thermal = 0
        dsim_rgb = 0
        if self.ssim:
            ssim_thermal = self.thermal_ssim(x_thermal, trans_thermal).detach()
            dsim_thermal = 1 - ssim_thermal

            ssim_rgb = self.rgb_ssim(x_rgb, trans_rgb).detach()
            dsim_rgb = 1 - ssim_rgb

        # cycle
        cycle_loss_thermal = 0
        cycle_loss_rgb = 0
        if self.cycle:
            cycle_rgb = self.gen(trans_thermal, mode='t2rgb')
            cycle_loss_rgb = self.recon_criterion(cycle_rgb, x_rgb).detach()

            cycle_thermal = self.gen(self.gen.gray_transformer(trans_rgb), mode='g2t')
            cycle_loss_thermal = self.recon_criterion(cycle_thermal, x_thermal).detach()

        thermal_loss = self.lambda_adv*thermal_adversarial_loss + trans_thermal_loss + thermal_edge_loss + dsim_thermal + cycle_loss_thermal
        rgb_loss = self.lambda_adv*gray_adversarial_loss + trans_rgb_loss + gray_rgb_loss + dsim_rgb + cycle_loss_rgb
        if val:
            return (thermal_loss + rgb_loss), (ssim_thermal+ssim_rgb)/2
        else:
            return (thermal_loss + rgb_loss)

    def _disc_step_thermal(self, x_gray, x_thermal):
        rec_thermal = self.gen(x_thermal, mode='t2t').detach()
        rec_thermal_logits = self.thermal_disc(rec_thermal)
        real_thermal_logits = self.thermal_disc(x_thermal)

        fake_rec_thermal_loss = self.adversarial_criterion(rec_thermal_logits, torch.zeros_like(rec_thermal_logits)).detach()
        real_thermal_loss = self.adversarial_criterion(real_thermal_logits, torch.ones_like(real_thermal_logits)).detach()

        trans_thermal = self.gen(x_gray, mode='g2t').detach()
        trans_thermal_logits = self.thermal_disc(trans_thermal)
        fake_trans_thermal_loss = self.adversarial_criterion(trans_thermal_logits, torch.zeros_like(trans_thermal_logits))
        loss = fake_rec_thermal_loss + fake_trans_thermal_loss + 2*real_thermal_loss
        return loss

    def _disc_step_rgb(self, x_gray, x_thermal, x_rgb):
        rec_rgb = self.gen(x_gray, mode='g2rgb').detach()
        rec_rgb_logits = self.color_disc(rec_rgb)
        real_rgb_logits = self.color_disc(x_rgb)

        fake_rec_rgb_loss = self.adversarial_criterion(rec_rgb_logits, torch.zeros_like(rec_rgb_logits)).detach()
        real_rgb_loss = self.adversarial_criterion(real_rgb_logits, torch.ones_like(real_rgb_logits)).detach()

        trans_rgb = self.gen(x_thermal, mode='t2rgb').detach()
        trans_rgb_logits = self.color_disc(trans_rgb)

        fake_trans_rgb_loss = self.adversarial_criterion(trans_rgb_logits, torch.zeros_like(trans_rgb_logits)).detach()

        loss = fake_rec_rgb_loss + fake_trans_rgb_loss + 2*real_rgb_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        # disc_params = list(self.thermal_disc.parameters()) + list(self.gray_disc.parameters())
        disc_gray_opt = torch.optim.Adam(self.color_disc.parameters(), lr=lr)
        disc_thermal_opt = torch.optim.Adam(self.thermal_disc.parameters(), lr=lr)
        return gen_opt, disc_gray_opt, disc_thermal_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)

        x_thermal = self.gen.resize(x_thermal)
        loss = None
        l2_reg = 0
        if optimizer_idx == 0:
            loss_gen_direct = self._gen_step_direct(x_gray, x_thermal, x_color)
            loss_gen_translate = self._gen_step_translate(x_gray, x_thermal, x_color)
            loss = loss_gen_direct + loss_gen_translate
            for param in self.gen.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Generator Loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            loss = self._disc_step_rgb(x_gray, x_thermal, x_color)
            for param in self.color_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train RGB Discriminator Loss", loss,  sync_dist=True)
        elif optimizer_idx == 2:
            loss = self._disc_step_thermal(x_gray, x_thermal)
            for param in self.thermal_disc.parameters():
                l2_reg += torch.norm(param)
            loss += self.l2_lambda * l2_reg
            self.log("Train Thermal Discriminator Loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_color, x_thermal, _, _ = batch
        x_color = self.gen.resize(x_color)
        x_gray = self.gen.gray_transformer(x_color)
        x_thermal = self.gen.resize(x_thermal)

        loss_gen_direct, ssim_direct = self._gen_step_direct(x_gray, x_thermal, x_color, val=True)
        loss_gen_translate, ssim_translate = self._gen_step_translate(x_gray, x_thermal, x_color, val=True)
        loss_gen = loss_gen_direct + loss_gen_translate
        self.log("Validation SSIM", (ssim_direct+ssim_translate)/2, sync_dist=True)
        self.log("Validation Generator Loss", loss_gen, sync_dist=True)
        return loss_gen

