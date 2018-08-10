import multiprocessing
import os
import pickle
import random
from multiprocessing import Pool

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

from config import alpha, identity_annot_filename, num_train_samples, num_valid_samples, cache_size, batch_size


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def get_indices():
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()

    ids = set()
    images = []
    image2id = {}
    id2images = {}

    for line in lines:
        line = line.strip()
        if len(line) > 0:
            tokens = line.split(' ')
            image_name = tokens[0].strip()
            id = tokens[1].strip()
            ids.add(id)
            images.append(image_name)
            image2id[image_name] = id
            if id in id2images.keys():
                id2images[id].append(image_name)
            else:
                id2images[id] = [image_name]

    return list(ids), images, image2id, id2images


def triplet_loss(y_true, y_pred):
    a_pred = y_pred[:, 0:128]
    p_pred = y_pred[:, 128:256]
    n_pred = y_pred[:, 256:384]
    positive_distance = K.square(tf.norm(a_pred - p_pred, axis=-1))
    negative_distance = K.square(tf.norm(a_pred - n_pred, axis=-1))
    loss = K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))
    return loss


def get_valid_triplets():
    ids, images, image2id, id2images = get_indices()
    num_samples = num_valid_samples
    images = images[num_train_samples:]

    data_set = []

    for i in tqdm(range(num_samples)):
        # choose a_image
        a_image = random.choice(images)
        a_id = image2id[a_image]
        while len(id2images[a_id]) <= 2:
            a_image = random.choice(images)
            a_id = image2id[a_image]
        # choose p_image
        p_image = random.choice(id2images[a_id])
        while p_image == a_image:
            p_image = random.choice(id2images[a_id])
        # choose n_image
        n_image = random.choice(images)
        while image2id[n_image] == image2id[a_image]:
            n_image = random.choice(images)

        data_set.append({'a': a_image, 'p': p_image, 'n': n_image})

    return data_set


def select_p_n_image(cache, a_image, image2id, id2images, embeddings, distance_mat, train_images):
    a_id = image2id[a_image]
    while len(id2images[a_id]) <= 2:
        a_image = random.choice(cache)
        a_id = image2id[a_image]
    a_index = cache.index(a_image)
    # choose p_image
    p_image = random.choice(id2images[a_id])
    while p_image == a_image or p_image not in train_images:
        p_image = random.choice(id2images[a_id])
    embedding_a = embeddings[a_image]
    embedding_p = embeddings[p_image]
    distance_a_p = np.square(np.linalg.norm(embedding_a - embedding_p))
    n_candidates = [q for q in range(cache_size) if
                    distance_mat[a_index, q] <= distance_a_p + alpha and distance_mat[
                        a_index, q] > distance_a_p and image2id[cache[q]] != a_id]
    if len(n_candidates) > 0:
        n_q = random.choice(n_candidates)
        return p_image, cache[n_q]
    else:
        return p_image, None


def select_one_batch(param):
    train_images, image2id, id2images, embeddings = param
    cache = sorted(random.sample(train_images, cache_size))
    distance_mat = np.empty(shape=(cache_size, cache_size), dtype=np.float32)
    batch_triplets = []
    for i in range(cache_size):
        for j in range(cache_size):
            embedding_i = embeddings[cache[i]]
            embedding_j = embeddings[cache[j]]
            dist = np.square(np.linalg.norm(embedding_i - embedding_j))
            distance_mat[i, j] = dist

    for j in range(batch_size):
        # choose a_image
        a_image = random.choice(cache)
        p_image, n_image = select_p_n_image(cache, a_image, image2id, id2images, embeddings, distance_mat,
                                            train_images)
        while n_image is None:
            a_image = random.choice(cache)
            p_image, n_image = select_p_n_image(cache, a_image, image2id, id2images, embeddings, distance_mat,
                                                train_images)
        batch_triplets.append({'a': a_image, 'p': p_image, 'n': n_image})
    return batch_triplets


def select_train_triplets():
    with open('data/train_embeddings.p', 'rb') as file:
        embeddings = pickle.load(file)

    ids, images, image2id, id2images = get_indices()
    train_images = images[:num_train_samples]
    num_batches = num_train_samples // batch_size
    train_triplets = []
    pool = Pool(20)
    params = []
    for _ in range(num_batches):
        params.append((train_images, image2id, id2images, embeddings))
    result = list(tqdm(pool.imap(select_one_batch, params), total=num_batches))
    for triplet_list in result:
        train_triplets.extend(triplet_list)

    return train_triplets


def get_train_images():
    _, images, _, _ = get_indices()
    return sorted(images[:num_train_samples])


def get_lfw_images():
    with open('data/people.txt', 'r') as file:
        lines = file.readlines()

    names = []

    for i in (range(2, len(lines))):
        line = lines[i].strip()
        tokens = line.split()
        if len(tokens) > 1:
            person_name = tokens[0]
            count = int(tokens[1])
            for j in range(1, count + 1):
                name = '{0}/{0}_{1}.jpg'.format(person_name, str(j).zfill(4))
                filename = os.path.join('data/lfw_funneled/', name)
                if os.path.isfile(filename):
                    names.append(name)
                else:
                    print(filename)

    return names


def get_pairs():
    with open('data/pairs.txt', 'r') as file:
        lines = file.readlines()

    pairs = []

    for i in (range(1, len(lines))):
        line = lines[i].strip()
        tokens = line.split()
        if len(tokens) == 3:
            person_name = tokens[0]
            id1 = int(tokens[1])
            id2 = int(tokens[2])
            image_name_1 = '{0}/{0}_{1}.jpg'.format(person_name, str(id1).zfill(4))
            image_name_2 = '{0}/{0}_{1}.jpg'.format(person_name, str(id2).zfill(4))
            pairs.append({'image_name_1': image_name_1, 'image_name_2': image_name_2, 'same_person': True})
        elif len(tokens) == 4:
            person_name_1 = tokens[0]
            id1 = int(tokens[1])
            person_name_2 = tokens[2]
            id2 = int(tokens[3])
            image_name_1 = '{0}/{0}_{1}.jpg'.format(person_name_1, str(id1).zfill(4))
            image_name_2 = '{0}/{0}_{1}.jpg'.format(person_name_2, str(id2).zfill(4))
            pairs.append({'image_name_1': image_name_1, 'image_name_2': image_name_2, 'same_person': False})

    return pairs


def get_latest_model():
    import glob
    import os
    files = glob.glob('models/*.hdf5')
    files.sort(key=os.path.getmtime)
    return files[-1]
