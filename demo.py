# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input

from config import img_size, channel, embedding_size, image_folder
from model import build_model
from utils import select_triplets

if __name__ == '__main__':
    model_weights_path = 'models/model.00-0.0296.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    num_samples = 10
    samples = select_triplets('valid')
    samples = random.sample(samples, num_samples)

    a_list = np.empty((10, 128), dtype=np.float32)
    p_list = np.empty((10, 128), dtype=np.float32)
    n_list = np.empty((10, 128), dtype=np.float32)

    for i in range(num_samples):
        sample = samples[i]
        batch_inputs = np.empty((3, 1, img_size, img_size, channel), dtype=np.float32)
        batch_dummy_target = np.zeros((1, embedding_size * 3), dtype=np.float32)

        for j, role in enumerate(['a', 'p', 'n']):
            image_name = sample[role]
            filename = os.path.join(image_folder, image_name)
            image_bgr = cv.imread(filename)
            image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            batch_inputs[j, 0] = preprocess_input(image_rgb)
            cv.imwrite('images/{}_{}_image.png'.format(i, role), image_bgr)

        y_pred = model.predict([batch_inputs[0], batch_inputs[1], batch_inputs[2]])
        a = y_pred[0, 0:128]
        p = y_pred[0, 128:256]
        n = y_pred[0, 256:384]

        a_list[i] = a
        p_list[i] = p
        n_list[i] = n

    with open('a_list.json', 'w') as file:
        json.dump(a_list.tolist(), file, indent=4)
    with open('p_list.json', 'w') as file:
        json.dump(p_list.tolist(), file, indent=4)
    with open('n_list.json', 'w') as file:
        json.dump(n_list.tolist(), file, indent=4)

    result = {}

    for i in range(num_samples):
        a_i = a_list[i]
        p_i = p_list[i]
        n_i = n_list[i]

        distance_a_p = np.linalg.norm(a_i - p_i) ** 2
        distance_a_n = np.linalg.norm(a_i - n_i) ** 2

        result['distance_{}_a_p'.format(i)] = distance_a_p
        result['distance_{}_a_n'.format(i)] = distance_a_n

        for j in range(num_samples):
            a_j = a_list[j]
            p_j = p_list[j]
            n_j = n_list[j]

            distance_i_j_a = np.linalg.norm(a_i - a_j) ** 2
            distance_i_j_p = np.linalg.norm(p_i - p_j) ** 2
            distance_i_j_n = np.linalg.norm(n_i - n_j) ** 2

            result['distance_{}_{}_a'.format(i, j)] = distance_i_j_a
            result['distance_{}_{}_p'.format(i, j)] = distance_i_j_p
            result['distance_{}_{}_n'.format(i, j)] = distance_i_j_n

    with open('result.json', 'w') as file:
        json.dump(result, file, indent=4)

    from replace_macro import replace

    replace()

    K.clear_session()
