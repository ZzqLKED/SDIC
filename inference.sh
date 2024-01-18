#python ./scripts/inference.py \
#--images_dir=./test_imgs  --n_sample=100 --edit_attribute='inversion'  \
#--save_dir=./experiment/inference_results  ./checkpoint/o_80000.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='lip'  \
# --save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='beard'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='eyes'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='smile' --edit_degree=1.0  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

 python /content/drive/MyDrive/AAAI24/SDIC/scripts/inference.py \
 --images_dir=/content/drive/MyDrive/AAAI24/HFGI-6/test_imgs --n_sample=100 --edit_attribute='pose' --edit_degree=2 \
 --save_dir=./experiment/inference_results  /content/drive/MyDrive/AAAI24/SDIC/checkpoint/SDIC_ffhq.pt #/content/drive/MyDrive/AAAI24/SDIC/experiment/ffhq/checkpoints/iteration_val_100000.pt
#