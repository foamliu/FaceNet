# encoding=utf-8
import json
import os

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence
import dlib
from augmentor import aug_pipe
from config import batch_size, img_size, channel, embedding_size, image_folder, lfw_folder, predictor_path


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

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictor_path)

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
                image = cv.imread(filename)     # BGR
                image = image[:, :, ::-1]       # RGB
                dets = self.detector(image, 1)

                num_faces = len(dets)
                if num_faces > 0:
                    # Find the 5 face landmarks we need to do the alignment.
                    faces = dlib.full_object_detections()
                    for detection in dets:
                        faces.append(self.sp(image, detection))

                    image = dlib.get_face_chip(image, faces[0], size=img_size)

                if self.usage == 'train':
                    image = aug_pipe.augment_image(image)

                batch_inputs[j, i_batch] = preprocess_input(image)

        return [batch_inputs[0], batch_inputs[1], batch_inputs[2]], batch_dummy_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)
