import pickle
import random
from multiprocessing import Pool
import math
import numpy as np
from tqdm import tqdm

from config import alpha, num_train_samples, cache_size, batch_size
from utils import get_data_stats

ids, images, image2id, id2images = get_data_stats('train')
train_images = images
with open('data/train_embeddings.p', 'rb') as file:
    embeddings = pickle.load(file)


def select_one_triplet(cache, distance_mat):
    # choose a_image
    while True:
        a_image = random.choice(cache)
        a_id = image2id[a_image]
        if len(id2images[a_id]) >= 2: break

    a_index = cache.index(a_image)

    # choose p_image
    while True:
        p_image = random.choice(id2images[a_id])
        if p_image != a_image and p_image in train_images: break

    # choose n_image
    # condition: distance_a_p <= distance_a_n <= distance_a_p + alpha
    embedding_a = embeddings[a_image]
    embedding_p = embeddings[p_image]
    distance_a_p = np.square(np.linalg.norm(embedding_a - embedding_p))
    n_candidates = [cache[n] for n in range(cache_size) if
                    distance_mat[a_index, n] <= distance_a_p + alpha and distance_mat[
                        a_index, n] > distance_a_p and image2id[cache[n]] != a_id]
    if len(n_candidates) > 0:
        n_image = random.choice(n_candidates)
        return a_image, p_image, n_image
    else:
        # argmin is a_index itself
        n_index = np.sort(distance_mat[a_index])[1]
        n_image = cache[n_index]
        return a_image, p_image, n_image


def select_one_batch(length):
    cache = sorted(random.sample(train_images, cache_size))
    distance_mat = np.empty(shape=(cache_size, cache_size), dtype=np.float32)
    for i in range(cache_size):
        for j in range(cache_size):
            embedding_i = embeddings[cache[i]]
            embedding_j = embeddings[cache[j]]
            dist = np.square(np.linalg.norm(embedding_i - embedding_j))
            distance_mat[i, j] = dist

    batch_triplets = []
    for j in range(length):
        a_image, p_image, n_image = select_one_triplet(cache, distance_mat)
        batch_triplets.append({'a': a_image, 'p': p_image, 'n': n_image})

    return batch_triplets


def select_train_triplets():
    num_batches = math.ceil(num_train_samples / batch_size)
    remain = num_train_samples
    batch_sizes = []
    for i in range(num_batches):
        if remain >= batch_size:
            batch_sizes.append(batch_size)
            remain -= batch_size
        else:
            batch_sizes.append(remain)

    pool = Pool(20)
    result = list(tqdm(pool.imap(select_one_batch, batch_sizes), total=num_batches))
    train_triplets = []
    for triplet_list in result:
        train_triplets.extend(triplet_list)

    return train_triplets
