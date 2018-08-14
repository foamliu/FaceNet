# encoding=utf-8
import json
import os

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence

from config import batch_size, img_size, channel, embedding_size, image_folder, lfw_folder


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        if self.usage == 'train':
            print('loading train samples')
            with open('data/train_triplets.json', 'r') as file:
                self.samples = json.load(file)
            self.image_folder = image_folder
        else:
            print('loading valid samples(LFW)')
            with open('data/lfw_val_triplets.json', 'r') as file:
                self.samples = json.load(file)
            self.image_folder = lfw_folder

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
                filename = os.path.join(self.image_folder, image_name)
                image_bgr = cv.imread(filename)
                image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
                if np.random.random_sample() > 0.5:
                    image_bgr = np.fliplr(image_bgr)
                image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
                batch_inputs[j, i_batch] = preprocess_input(image_rgb)

        return [batch_inputs[0], batch_inputs[1], batch_inputs[2]], batch_dummy_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)
