# encoding=utf-8
import os

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence

from config import batch_size, img_size, channel, embedding_size, image_folder
from utils import select_triplets


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        self.samples = select_triplets(usage)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_inputs = np.empty((3, length, img_size, img_size, channel), dtype=np.float32)
        batch_dummy_target = np.zeros((length, embedding_size * 3), dtype=np.float32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            for j, role in enumerate(['a', 'p', 'n']):
                image_name = sample[role]
                filename = os.path.join(image_folder, image_name)
                image_bgr = cv.imread(filename)
                image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
                image_rgb = cv.resize(image_rgb, (img_size, img_size), cv.INTER_CUBIC)
                batch_inputs[j, i_batch] = image_rgb

        for j in range(3):
            batch_inputs[j] = preprocess_input(batch_inputs[j])

        return [batch_inputs[0], batch_inputs[1], batch_inputs[2]], batch_dummy_target

    def on_epoch_end(self):
        self.samples = select_triplets(self.usage)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
