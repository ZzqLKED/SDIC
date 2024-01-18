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
from models.stylegan2.model import Generator


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'car' in opts.dataset_type
    generatoro = net.decoder
    generatoro.eval()
    aligner = net.CGI
    args, data_loader = setup_data_loader(args, opts)
    editor = latent_editor.LatentEditor(net.decoder, is_cars)
    generator = Generator(1024, 512, 8, channel_multiplier=2).cuda()
    checkpoint = torch.load(args.finetuned_generator_checkpoint_path)
    generator.load_state_dict(checkpoint['g_ema'])

    # initial inversion
    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

    edit_directory_path = os.path.join(args.save_dir, args.edit_attribute)
    os.makedirs(edit_directory_path, exist_ok=True)

    # perform high-fidelity inversion or editing
    for i, batch in enumerate(data_loader):
        if args.n_sample is not None and i > args.n_sample:
            print('inference finished!')
            break            
        x = batch.to(device).float()

        #'''
        # calculate the distortion map
        imgs, latents_ ,G7= generator([latent_codes[i].unsqueeze(0).to(device)],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        
        res_align = net.CGI(x,torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear'))
        F,Wp= net.residue(res_align,G7,latents_)
        
        imgs, lante_, G_ = generator([latents_],G7, input_is_latent=True, randomize_noise=False, return_latents=True)

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

def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'domain_adaptation_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'domain_adaptation_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)

    restyle_e4e, restyle_opts = load_model(test_opts.restyle_checkpoint_path,
                                           update_opts={"resize_outputs": test_opts.resize_outputs,
                                                        "n_iters_per_batch": test_opts.restyle_n_iterations},
                                           is_restyle_encoder=True)
    finetuned_generator = load_generator(test_opts.finetuned_generator_checkpoint_path)

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    for input_batch in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            result_batch, _ = run_domain_adaptation(input_cuda, net, opts, finetuned_generator,
                                                    restyle_e4e, restyle_opts)

        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]

            curr_result = tensor2im(result_batch[i])
            input_im = tensor2im(input_batch[i])

            res_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            curr_result.resize(resize_amount).save(res_save_path)

            coupled_save_path = os.path.join(out_path_coupled, os.path.basename(im_path))
            res = np.concatenate([np.array(input_im.resize(resize_amount)), np.array(curr_result.resize(resize_amount))],
                                 axis=1)
            Image.fromarray(res).save(coupled_save_path)
            global_i += 1


if __name__ == "__main__":

    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
    parser.add_argument('--finetuned_generator_checkpoint_path', type=str, default=model_paths["stylegan_pixar"],
                                 help='Path to fine-tuned generator checkpoint used for domain adaptation.')
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")
    parser.add_argument("--stylegan_weights", type=str, default="/home/jrq/zzq/HFGI-5/pretrained/stylegan2-ffhq-config-f.pt")
    
    args = parser.parse_args()
    main(args)