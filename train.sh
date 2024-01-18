python ./scripts/train.py   --dataset_type='ffhq_encode'  --start_from_latent_avg \
--id_lambda=0.1  --val_interval=40000 --save_interval=40000 --max_steps=200100  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0   \
--stylegan_weights='./pretrained/stylegan2-ffhq-config-f.pt' --checkpoint_path='./pretrained/e4e_ffhq_encode.pt'  \
--workers=48  --batch_size=2  --test_batch_size=4  --test_workers=48 --exp_dir='./experiment/ffhq'  

#--optim_name='adam' 