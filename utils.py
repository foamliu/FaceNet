import multiprocessing
import os

import cv2 as cv
import keras.backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import alpha


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
