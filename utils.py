import multiprocessing
import os
import random

import cv2 as cv
import keras.backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

from config import alpha, identity_annot_filename, bbox_annot_filename


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


def triplet_loss(y_true, y_pred):
    a_pred = y_pred[:, 0:128]
    p_pred = y_pred[:, 128:256]
    n_pred = y_pred[:, 256:384]
    loss = K.mean(
        tf.norm(a_pred - p_pred, ord='euclidean', axis=-1) - tf.norm(a_pred - n_pred, ord='euclidean', axis=-1)) + alpha
    return loss


def get_bbox():
    with open(bbox_annot_filename, 'r') as file:
        lines = file.readlines()

    image2bbox = {}
    for i in range(2, len(lines)):
        line = lines[i]
        line = line.strip()
        if len(line) > 0:
            tokens = line.split()
            image_name = tokens[0]
            x_1 = int(tokens[1])
            y_1 = int(tokens[2])
            width = int(tokens[3])
            height = int(tokens[4])
            image2bbox[image_name] = (x_1, y_1, width, height)
    return image2bbox


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


def select_triplets(num_samples):
    ids, images, image2id, id2images = get_indices()
    data_set = []

    for i in tqdm(range(num_samples)):
        a_id = random.choice(ids)
        while len(id2images[a_id]) <= 2:
            a_id = random.choice(ids)
        a_p_images = random.sample(id2images[a_id], 2)
        a_image = a_p_images[0]
        p_image = a_p_images[1]
        n_image = random.choice(images)
        data_set.append({'a': a_image, 'p': p_image, 'n': n_image})

    return data_set
