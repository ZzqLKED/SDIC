import torch
import argparse
from models.SDIC import SDIC
from models.stylegan2.model import Generator

def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    is_cars = 'car' in opts['dataset_type']
    is_faces = ('ffhq' in opts['dataset_type'] )or ('celeba' in opts['dataset_type'])
    #is_faces = 

    if is_faces:
        opts['stylegan_size'] = 1024
    elif is_cars:
        opts['stylegan_size'] = 512
    else:
        opts['stylegan_size'] = 256

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts['is_train'] = False
    opts = argparse.Namespace(**opts)

    net = SDIC(opts)
    net.eval()
    net = net.to(device)
    return net, opts
    
def load_generator(checkpoint_path, device='cuda'):
    print(f"Loading generator from checkpoint: {checkpoint_path}")
    generator = Generator(1024, 512, 8, channel_multiplier=2)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(ckpt['g_ema'])
    generator.eval()
    generator.to(device)
    return generator
