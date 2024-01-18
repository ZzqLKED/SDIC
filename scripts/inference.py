import argparse
import torch
import numpy as np
import sys
import os

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from PIL import Image
import pickle
from editings import latent_editor
from configs.paths_config import edit_paths, model_paths
from editings.styleclip.global_direction import StyleCLIPGlobalDirection
from editings.styleclip.model import Generator

from torchvision import utils
import time

def load_direction_calculator(args):
    delta_i_c = torch.from_numpy(np.load(args.delta_i_c)).float().cuda()
    with open(args.s_statistics, "rb") as channels_statistics:
        _, s_std = pickle.load(channels_statistics)
        s_std = [torch.from_numpy(s_i).float().cuda() for s_i in s_std]
    with open(args.text_prompt_templates, "r") as templates:
        text_prompt_templates = templates.readlines()
    global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates)
    return global_direction_calculator


def main(args):
    net, opts = setup_model(args.ckpt, device)
##########
    '''
    save_name =  'SDIC_ffhq.pt'
    save_dict = {
            'state_dict': net.state_dict(),
            'opts': vars(opts)}
    save_dict['latent_avg'] = net.latent_avg
    
    checkpoint_path = os.path.join('/content/drive/MyDrive/AAAI24/SDIC/checkpoint', save_name)
    torch.save(save_dict, checkpoint_path)
    '''
##########
    is_cars = 'car' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    aligner = net.DIP
    args, data_loader = setup_data_loader(args, opts)
    editor = latent_editor.LatentEditor(net.decoder, is_cars)
    global_direction_calculator = load_direction_calculator(args)
    stylegan_model = Generator(1024, 512, 8, channel_multiplier=2).cuda()
    checkpoint = torch.load(args.stylegan_weights)
    stylegan_model.load_state_dict(checkpoint['g_ema'])

    # initial inversion
    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

    # set the editing operation
    if args.edit_attribute == 'inversion':
        pass
    elif args.edit_attribute == 'clip':
        pass
    elif args.edit_attribute == 'age' or args.edit_attribute == 'smile' or args.edit_attribute == 'pose':
        interfacegan_directions = {
                'age': './editings/interfacegan_directions/age.pt',
                'smile': './editings/interfacegan_directions/smile.pt',
                'pose': './editings/interfacegan_directions/pose.pt' }
        edit_direction = torch.load(interfacegan_directions[args.edit_attribute]).to(device)
    elif args.edit_attribute == 'eyes'or args.edit_attribute == 'beard' or args.edit_attribute == 'lip':
        ganspace_pca = torch.load('./editings/ganspace_pca/ffhq_pca.pt') 
        ganspace_directions = {
            'eyes':            (54,  7,  8,  20),
            'beard':           (58,  7,  9,  -20),
            'lip':             (34, 10, 11,  20) }            
        edit_direction = ganspace_directions[args.edit_attribute]
    else :
        ganspace_pca = torch.load('./editings/ganspace_pca/cars_pca.pt') 
        ganspace_directions = {
            'left':            (0,  0,  5,  2),
            'right':           (0,  0,  5,  -2),
            'cube':             (16, 3, 6,  25),
            'color':            (22, 9, 11, -8),
            'grass':             (41, 9, 11, -18)
        }
        edit_direction = ganspace_directions[args.edit_attribute]

    edit_directory_path = os.path.join(args.save_dir, args.edit_attribute)
    os.makedirs(edit_directory_path, exist_ok=True)

    # perform high-fidelity inversion or editing
    for i, batch in enumerate(data_loader):
        if args.n_sample is not None and i > args.n_sample:
            print('inference finished!')
            break            
        x = batch.to(device).float()
        t1 = time.time()
        # calculate the distortion map
        if args.edit_attribute == 'clip':
            imgs, latents_ ,latents,G7= stylegan_model( styles=[latent_codes[i].unsqueeze(0).to(device)],
                                                        G=None, 
                                                        input_is_latent=True,
                                                        randomize_noise=False,
                                                        return_latents=True,
                                                        truncation=1,
                                                        truncation_latent=None)
        else :
            imgs, latents_ ,G7= generator([latent_codes[i].unsqueeze(0).to(device)],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        

        intial = tensor2im(imgs[0])
        
        #x=torch.nn.functional.interpolate(torch.clamp(x, -1., 1.), size=(256,256) , mode='bilinear')
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear') #face size=256 256
        
        
        res_align = net.DIP(x,torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')) #face size=256 256


        F,Wp= net.SAF(res_align,G7,latents_)
        # produce initial editing image
        
        if args.edit_attribute == 'inversion':
            img_edit = imgs
            edit_latents = Wp#latent_codes[i].unsqueeze(0).to(device)
            Ge = G7
        elif args.edit_attribute == 'clip':
            alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alphas)
            betas = np.linspace(args.beta_min, args.beta_max, args.num_betas)
            results = []
            #Wp=torch.squeeze(Wp,0)
            #print(Wp.shape)
            for beta in betas:
                direction = global_direction_calculator.get_delta_s(args.neutral_text, args.target_text, beta)
                edit_latents = [[s_i + alpha * b_i for s_i, b_i in zip(latents, direction)] for alpha in alphas]
                edit_latents = [torch.cat([edit_latents[i][j] for i in range(args.num_alphas)])
                                        for j in range(len(edit_latents[0]))]
                for b in range(0, edit_latents[0].shape[0]):
                    edit_latents_batch = [s_i[b:b + 1] for s_i in edit_latents]
                    img_edit, edited_latents,edit_latentv, Ge = stylegan_model([edit_latents_batch],
                                                        None,
                                                        input_is_stylespace=True,
                                                        randomize_noise=False,
                                                        return_latents=True)
        elif args.edit_attribute == 'age' or args.edit_attribute == 'smile' or args.edit_attribute == 'pose':
            img_edit, Ge, edit_latents = editor.apply_interfacegan( Wp, edit_direction, factor=args.edit_degree)
        else:
            img_edit, Ge, edit_latents = editor.apply_ganspace(Wp, ganspace_pca, [edit_direction])

        F = F - G7 + Ge

        if args.edit_attribute == 'clip':
            imgs, lante_,_, G_ = stylegan_model([edit_latentv],
                                                        G=F,
                                                        input_is_stylespace=True,
                                                        randomize_noise=False,
                                                        return_latents=True)
        else :
            imgs, lante_, G_ = generator([edit_latents],F, input_is_latent=True, randomize_noise=False, return_latents=True)
        
        t2 =time.time()
        print(float(t2-t1))
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
            
        # save images
        result = tensor2im(imgs[0])
        im_save_path = os.path.join(edit_directory_path, f"{i:05d}.jpg")
        Image.fromarray(np.array(result)).save(im_save_path)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)

if __name__ == "__main__":

    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--beta_min", type=float, default=0.10)#0.11
    parser.add_argument("--beta_max", type=float, default=0.16)#0.16
    parser.add_argument("--num_betas", type=int, default=5)#5
    parser.add_argument("--alpha_min", type=float, default=-5)#-5
    parser.add_argument("--alpha_max", type=float, default=5)#5
    parser.add_argument("--num_alphas", type=int, default=11) #11
    parser.add_argument("--neutral_text", type=str, default="face with hair")
    parser.add_argument("--target_text", type=str, default="face with red hair")
    parser.add_argument("--delta_i_c", type=str, default=edit_paths["styleclip"]["delta_i_c"],
                        help="path to file containing delta_i_c")
    parser.add_argument("--s_statistics", type=str, default=edit_paths["styleclip"]["s_statistics"],
                        help="path to file containing s statistics")
    parser.add_argument("--text_prompt_templates", default=edit_paths["styleclip"]["templates"])
    parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")
    parser.add_argument("--stylegan_weights", type=str, default="./pretrained/stylegan2-ffhq-config-f.pt")
    

    args = parser.parse_args()
    main(args)