import json
import math
import pickle
import random
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from config import alpha, num_train_samples, triplets_selection_batch_size, semi_hard_mode, hard_mode
from utils import get_data_stats

ids, images, image2id, id2images = get_data_stats()
train_images = images
with open('data/train_embeddings.p', 'rb') as file:
    embeddings = pickle.load(file)


def select_hard(batch, distance_mat, a_image, a_id, p_image):
    a_index = batch.index(a_image)
    # condition: argmin(distance_a_n)
    indices = np.argsort(distance_mat[a_index])
    for n_index in indices:
        n_image = batch[n_index]
        if n_image != a_image and n_image != p_image and image2id[n_image] != a_id:
            break

    return n_image


def select_semi_hard(batch, distance_mat, a_image, a_id, p_image):
    a_index = batch.index(a_image)

    embedding_a = embeddings[a_image]
    embedding_p = embeddings[p_image]
    distance_a_p = np.square(np.linalg.norm(embedding_a - embedding_p))

    length = len(batch)

    # condition: distance_a_p <= distance_a_n <= distance_a_p + alpha
    n_candidates = [batch[n] for n in range(length) if
                    image2id[batch[n]] != a_id and distance_mat[a_index, n] <= distance_a_p + alpha and distance_mat[
                        a_index, n] > distance_a_p]
    if len(n_candidates) == 0:
        # if not found, loose condition: distance_a_n <= distance_a_p + alpha
        n_candidates = [batch[n] for n in range(length) if
                        image2id[batch[n]] != a_id and distance_mat[a_index, n] <= distance_a_p + alpha]
        if len(n_candidates) == 0:
            # if still not found, select hard.
            # n_candidates = [batch[n] for n in range(length) if image2id[batch[n]] != a_id]
            n_image = select_hard(batch, distance_mat, a_image, a_id, p_image)
            n_candidates = [n_image]

    n_image = random.choice(n_candidates)
    return n_image


def select_one_triplet(batch, a_index, distance_mat, select_mode):
    # choose a_image
    a_image = batch[a_index]
    a_id = image2id[a_image]
    if len(id2images[a_id]) < 2:
        raise ValueError('Cannot find any positives for the specified anchor image.')

    # choose p_image
    p_image = random.choice([p for p in id2images[a_id] if p != a_image])

    # choose n_image
    if select_mode == semi_hard_mode:
        n_image = select_semi_hard(batch, distance_mat, a_image, a_id, p_image)
    else:
        n_image = select_hard(batch, distance_mat, a_image, a_id, p_image)

    return a_image, p_image, n_image


def select_one_batch(config):
    start, end, select_mode = config
    length = end - start
    batch = train_images[start:end]
    distance_mat = np.empty(shape=(length, length), dtype=np.float32)
    for i in range(length):
        for j in range(length):
            embedding_i = embeddings[batch[i]]
            embedding_j = embeddings[batch[j]]
            dist = np.square(np.linalg.norm(embedding_i - embedding_j))
            distance_mat[i, j] = dist

    batch_triplets = []
    for a_index in range(length):
        try:
            a_image, p_image, n_image = select_one_triplet(batch, a_index, distance_mat, select_mode)
            batch_triplets.append({'a': a_image, 'p': p_image, 'n': n_image})
        except ValueError:
            pass

    return batch_triplets


def select_train_triplets(select_mode):
    num_batches = int(math.ceil(num_train_samples / triplets_selection_batch_size))
    remain = num_train_samples
    batch_configs = []
    for i in range(num_batches):
        start = i * triplets_selection_batch_size
        if remain >= triplets_selection_batch_size:
            end = start + triplets_selection_batch_size
            remain -= triplets_selection_batch_size
        else:
            end = start + remain
        batch_configs.append((start, end, select_mode))

    pool = Pool(20)
    result = list(tqdm(pool.imap(select_one_batch, batch_configs), total=num_batches))
    train_triplets = []
    for triplet_list in result:
        train_triplets.extend(triplet_list)

    return train_triplets


if __name__ == '__main__':
    np.random.shuffle(train_images)
    train_triplets = select_train_triplets(hard_mode)

    with open('data/train_triplets.json', 'w') as file:
        json.dump(train_triplets, file)

    print('len(train_triplets): ' + str(len(train_triplets)))
