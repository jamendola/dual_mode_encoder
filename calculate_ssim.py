import kaist
from kaist import get_data, Kaist
import numpy as np
import torch
from ssim import SSIM

from light_gan import LightFlexCondGAN, CycleLES4G2DEncRegCondGAN, LES2DCondGAN, CycleLES4G2DEncRegCondGAN, \
    VGGES4G2DCondGAN, CLE2SR2dualG2DCondGAN

input_path = "../kaist/kaist-cvpr15"
models = list()
# imgs_dicts = get_data(input_path, mode="train")
ds = Kaist(input_path, set_name="test_cut")
from torchvision.transforms import Normalize

thermal_normalizer = ds.thermal_normalizer
thermal_normalizer.mean
thermal_unnormalize = Normalize((-thermal_normalizer.mean / thermal_normalizer.std), (1.0 / thermal_normalizer.std))
color_normalizer = ds.color_normalizer
color_normalizer.mean
color_unnormalize = Normalize((-color_normalizer.mean / color_normalizer.std).tolist(), (1.0 / color_normalizer.std).tolist())



model_higher_adv = LightFlexCondGAN(lambda_adv=0.1, cycle=True, edge=True, ssim=True)
model_higher_adv = model_higher_adv.load_from_checkpoint("trained_models/03.30.2023-20.40.23unnorm_benchmark/debug.ckpt")
ae_model = model_higher_adv.gen
models.append(ae_model)

#
# unnorm_bench = LightFlexCondGAN(lambda_adv=0.05, cycle=True, edge=True, ssim=True)
# unnorm_bench = unnorm_bench.load_from_checkpoint(
#     "trained_models/04.02.2023-11.48.12norm_benchmark/debug_200.ckpt")
# ae_model = unnorm_bench.gen
# models.append(ae_model)
#
# ebit = LightFlexCondGAN(lambda_adv=0.05, cycle=True, edge=True, ssim=True)
# ebit = ebit.load_from_checkpoint(
#     "trained_models/04.02.2023-11.48.12ebit/debug_200.ckpt")
# ae_model = ebit.gen
# models.append(ae_model)
#
# patel = LightFlexCondGAN(lambda_adv=0.05, cycle=True, edge=True, ssim=True)
# patel = patel.load_from_checkpoint(
#     "trained_models/04.02.2023-11.48.12patel/debug_200.ckpt")
# ae_model = patel.gen
# models.append(ae_model)
#
# model = LightFlexCondGAN(lambda_adv=0.05, cycle=True, edge=True, ssim=True)
# model = model.load_from_checkpoint(
#     "trained_models/04.02.2023-11.48.12luo_e/debug_200.ckpt")
# ae_model = model.gen
# models.append(ae_model)
#
# model = LightFlexCondGAN(lambda_adv=0.05, cycle=True, edge=True, ssim=True)
# model = model.load_from_checkpoint(
#     "trained_models/04.02.2023-11.48.12explore_g/debug_200.ckpt")
# ae_model = model.gen
# models.append(ae_model)
#
# model = LightFlexCondGAN(lambda_adv=0.05, cycle=True, edge=True, ssim=True)
# model = model.load_from_checkpoint(
#     "trained_models/04.02.2023-11.48.12explore_f/debug_200.ckpt")
# ae_model = model.gen
# models.append(ae_model)

ssim_rgb = SSIM(3)
ssim_t = SSIM(3)

length = len(ds)
# length = 100
error = torch.zeros(length, 4)

for row, model in enumerate(models):
    print('model', row)
    for index in range(length):
        #         print('value',index)
        X_rgb = ds[index][0].float()
        X_rgb = model.resize(X_rgb.unsqueeze(0))
        X_t = ds[index][1].float()
        X_t = model.resize(X_t.unsqueeze(0))
        X_gray = model.gray_transformer(X_rgb)
        model.eval()
        # print('Direct mode ', modes[mode][0])

        # if row == 6:
        #     X_g2rgb, _ = model(X_gray, mode='rgb')
        #     X_g2t, _ = model(X_gray, mode='thermal')
        #     X_t2t, _ = model(X_t, mode='thermal')
        #     X_t2rgb, _ = model(X_t, mode='rgb')
        #
        # else:
        X_g2rgb = model(X_gray, mode='g2rgb')
        X_g2t = model(X_gray, mode='g2t')
        X_t2t = model(X_t, mode='t2t')
        X_t2rgb = model(X_t, mode='t2rgb')

        s_g2rgb = ssim_rgb(X_rgb, X_g2rgb)
        s_t2rgb = ssim_rgb(X_rgb, X_t2rgb)
        s_g2t = ssim_t(X_t, X_g2t)
        s_t2t = ssim_t(X_t, X_t2t)
        #         print([s_g2rgb, s_t2rgb, s_g2t,s_t2t])
        error[index, :] = torch.tensor([s_g2rgb, s_t2rgb, s_g2t, s_t2t])

    print(torch.mean(error, 0))
