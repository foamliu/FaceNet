# encoding=utf-8
import json
import multiprocessing as mp
import os
from multiprocessing import Pool

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm

from config import img_size, identity_annot_filename, image_folder, num_images
from model import build_model

model_weights_path = 'models/model.00-0.0296.hdf5'
model = build_model()
model.load_weights(model_weights_path)


def inference_one_image(item):
    image_name_0, image_name_1, image_name_2, out_queue = item

    filename = os.path.join(image_folder, image_name_0)
    image_bgr = cv.imread(filename)
    image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    image_rgb_0 = preprocess_input(image_rgb)
    filename = os.path.join(image_folder, image_name_1)
    image_bgr = cv.imread(filename)
    image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    image_rgb_1 = preprocess_input(image_rgb)
    filename = os.path.join(image_folder, image_name_2)
    image_bgr = cv.imread(filename)
    image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    image_rgb_2 = preprocess_input(image_rgb)

    batch_inputs = np.empty((3, 1, img_size, img_size, 3), dtype=np.float32)
    batch_inputs[0] = image_rgb_0
    batch_inputs[1] = image_rgb_1
    batch_inputs[2] = image_rgb_2
    y_pred = model.predict([batch_inputs[0], batch_inputs[1], batch_inputs[2]])

    a = y_pred[0, 0:128]
    p = y_pred[0, 128:256]
    n = y_pred[0, 256:384]

    out_queue.put({'image_name': image_name_0, 'embedding': a})
    out_queue.put({'image_name': image_name_1, 'embedding': p})
    out_queue.put({'image_name': image_name_2, 'embedding': n})


def inference():
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()

    pool = Pool(24)
    manager = mp.Manager()
    out_queue = manager.Queue()

    items = []
    for i in range(num_images // 3):
        image_names_0 = (lines[3 * i].split(' ')[0].strip())
        image_names_1 = (lines[3 * i + 1].split(' ')[0].strip())
        image_names_2 = (lines[3 * i + 2].split(' ')[0].strip())
        items.append((image_names_0, image_names_1, image_names_2, in_queue, out_queue))

    for _ in tqdm(pool.imap_unordered(inference_one_image, items), total=len(lines)):
        pass
    pool.close()
    pool.join()

    out_list = []
    while out_queue.qsize() > 0:
        out_list.append(out_queue.get())

    with open("data/preds.p", "w") as file:
        json.dump(out_list, file, indent=4)


if __name__ == '__main__':
    inference()
