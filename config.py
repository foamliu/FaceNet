img_size = 139
channel = 3
batch_size = 256
epochs = 10000
patience = 10
embedding_size = 128
num_images = 202599
num_identities = 10177
valid_ratio = 0.005
num_train_samples = 200001
num_valid_samples = 2598

alpha = 0.2
SENTINEL = 1

image_folder = 'data/img_align_celeba'
identity_annot_filename = 'data/identity_CelebA.txt'
bbox_annot_filename = 'data/list_bbox_celeba.txt'
lfw_folder = 'data/lfw_funneled'

semi_hard_mode = 'semi-hard'
hard_mode = 'hard'
triplet_select_mode = hard_mode

best_model = 'models/model.01-0.0087.hdf5'

