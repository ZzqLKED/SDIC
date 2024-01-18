from criteria.lpips.lpips import LPIPS
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.SDIC import SDIC
from training.ranger import Ranger
from argparse import Namespace


from torchvision.utils import save_image
import os
import PIL
import PIL.Image as Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms
from utils.common import tensor2im



EXPERIMENT_ARGS = {}
EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
resize_dims = (256, 256)


array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
# this loop is for read each image in this foder,directory_name is the foder name with images.
for filename in os.listdir(r"./experment/value"):
    array_of_img.append(filename)
print(array_of_img)

transf = transforms.ToTensor()
lpips_loss = LPIPS(net_type='alex').to("cuda").eval()
mse_loss = nn.MSELoss()
lpip = 0
lpsnr =0
lssim =0
lmse =0
num = 0
model_path = "./experment/checkpoints/iteration_val_100000.pt"
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['is_train'] = False
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')


def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0,
 :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


for image in array_of_img:
  image_path = './experment/value/'+image
  original_image = Image.open(image_path)
  input_image = original_image.convert("RGB")
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)

  with torch.no_grad():
    x = transformed_image.unsqueeze(0).cuda()

    latent_codes = get_latents(net, x)

    y_hat, result_latent, delta, imgs_, loss_res,F,W = net.forward(x, return_latents=True)

    res = tensor2im(y_hat[0])
    res.save(os.path.join('./experment/output/',image))
    if num%10 == 0:
        print(num," is ok! ")
    num+=1