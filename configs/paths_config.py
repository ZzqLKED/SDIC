dataset_paths = {
	#  Face Datasets (FFHQ - train, CelebA-HQ - test)
	'ffhq': '/home/jrq/zzq/data/images256x256',
	'ffhq_val': '/home/jrq/zzq/data/celeba1024/test',

	#  Cars Dataset (Stanford cars)
	'cars_train': '/home/jrq/zzq/data/car',
	'cars_val': '/home/jrq/zzq/data/car/cars_test',

	# CelebA-HQ(25000train 5000test)
	'celeba': '/home/jrq/zzq/data/celeba1024/train',
	'celeba_val': '/home/jrq/zzq/data/celeba1024/test',
}

model_paths = {
	'stylegan_ffhq': './pretrained/stylegan2-ffhq-config-f.pt',
	'ir_se50': './pretrained/model_ir_se50.pth',
	'shape_predictor': './pretrained/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained/moco_v2_800ep_pretrain.pt',
	'stylegan_pixar': './pretrained/pixar.pt'
}

edit_paths = {
	'age': 'editing/interfacegan_directions/age.pt',
	'smile': 'editing/interfacegan_directions/smile.pt',
	'pose': 'editing/interfacegan_directions/pose.pt',
	'cars': 'editing/ganspace_directions/cars_pca.pt',
	'styleclip': {
		'delta_i_c': 'editings/styleclip/global_directions/ffhq/fs3.npy',
		's_statistics': 'editings/styleclip/global_directions/ffhq/S_mean_std',
		'templates': 'editings/styleclip/global_directions/templates.txt'
	}
}