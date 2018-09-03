img_size = 139
channel = 3
batch_size = 128
epochs = 10
patience = 10
embedding_size = 128
num_images = 202599
num_identities = 10177
valid_ratio = 0.005
# 5,600 were excluded as they cannot be aligned by dlib
# 202,599 - 5,600 = 196,999, separate into two classes: train and valid.
num_celeba_train_samples = 194814
num_celeba_valid_samples = 2185
num_lfw_valid_samples = 2185    # LFW data set: 6000 pairs => 2185 triplets
predictor_path = 'models/shape_predictor_5_face_landmarks.dat'
alpha = 0.2
SENTINEL = 1
threshold = 0.8

image_folder = 'data/img_align_celeba'
identity_annot_filename = 'data/identity_CelebA.txt'
bbox_annot_filename = 'data/list_bbox_celeba.txt'
lfw_folder = 'data/lfw_funneled'

semi_hard_mode = 'semi-hard'
hard_mode = 'hard'
triplet_select_mode = semi_hard_mode

best_model = 'models/model.01-0.0087.hdf5'
