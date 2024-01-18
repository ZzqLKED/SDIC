python scripts/run_domain_adaptation.py \
--images_dir=/home/jrq/zzq/HFGI-5/test/face \
--n_sample=100 --edit_attribute='sketch' \
--finetuned_generator_checkpoint_path=pretrained/sketch_hq.pt \
--save_dir=./experiment/inference_results /home/jrq/zzq/HFGI-5/experiment/ffhq_sig/checkpoints/iteration_val_100000.pt 
