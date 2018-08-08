# encoding=utf-8
# import the necessary packages
import json
import multiprocessing as mp
import os
import queue
from multiprocessing import Process

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm

from config import img_size, image_folder, identity_annot_filename, num_images


class InferenceWorker(Process):
    def __init__(self, gpuid, in_queue, out_queue, signal_queue):
        Process.__init__(self, name='ImageProcessor')

        self.gpuid = gpuid
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.signal_queue = signal_queue

    def run(self):
        # set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        print("InferenceWorker init, GPU ID: {}".format(self.gpuid))

        from model import build_model

        # load models
        model_weights_path = 'models/model.00-0.0296.hdf5'
        model = build_model()
        model.load_weights(model_weights_path)

        while True:
            try:
                try:
                    item = self.in_queue.get(block=False)
                except queue.Empty:
                    continue

                image_name_0, image_name_1, image_name_2 = item

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

                self.out_queue.put({'image_name': image_name_0, 'embedding': a})
                self.out_queue.put({'image_name': image_name_1, 'embedding': p})
                self.out_queue.put({'image_name': image_name_2, 'embedding': n})
                if self.in_queue.qsize() == 0:
                    break
            except Exception as e:
                print(e)

        import keras.backend as K
        K.clear_session()
        print('InferenceWorker done, GPU ID {}'.format(self.gpuid))


class Scheduler:
    def __init__(self, gpuids, signal_queue):
        self.signal_queue = signal_queue
        self.in_queue = manager.Queue()
        self.out_queue = manager.Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(InferenceWorker(gpuid, self.in_queue, self.out_queue, self.signal_queue))

    def start(self, names):
        # put all of image names into queue
        for name in names:
            self.in_queue.put(name)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")
        return self.out_queue


def run(gpuids, q):
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()

    items = []
    for i in range(num_images // 3):
        image_names_0 = (lines[3 * i].split(' ')[0].strip())
        image_names_1 = (lines[3 * i + 1].split(' ')[0].strip())
        image_names_2 = (lines[3 * i + 2].split(' ')[0].strip())
        items.append((image_names_0, image_names_1, image_names_2))

    # init scheduler
    x = Scheduler(gpuids, q)

    # start processing and wait for complete
    return x.start(items)


SENTINEL = 1


def listener(q):
    pbar = tqdm(total=num_images)
    for item in iter(q.get, None):
        pbar.update()


if __name__ == "__main__":
    gpuids = ['0', '1', '2', '3']
    print(gpuids)

    manager = mp.Manager()
    q = manager.Queue()
    proc = mp.Process(target=listener, args=(q,))
    proc.start()

    out_queue = run(gpuids, q)
    out_list = []
    while out_queue.qsize() > 0:
        out_list.append(out_queue.get())

    with open("data/preds.p", "w") as file:
        json.dump(out_list, file, indent=4)

    q.put(None)
    proc.join()
