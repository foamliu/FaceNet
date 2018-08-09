import multiprocessing as mp
import os
import pickle
import queue
from multiprocessing import Process
from multiprocessing import Process

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm

from config import best_model, lfw_folder, img_size, channel
from utils import get_lfw_images, get_pairs


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
        model = build_model()
        model.load_weights(best_model)

        while True:
            try:
                sample = {}
                try:
                    sample['a'] = self.in_queue.get(block=False)
                    sample['p'] = self.in_queue.get(block=False)
                    sample['n'] = self.in_queue.get(block=False)

                except queue.Empty:
                    continue

                batch_inputs = np.empty((3, 1, img_size, img_size, channel), dtype=np.float32)

                for j, role in enumerate(['a', 'p', 'n']):
                    image_name = sample[role]
                    filename = os.path.join(lfw_folder, image_name)
                    image_bgr = cv.imread(filename)
                    image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
                    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
                    batch_inputs[j, 0] = preprocess_input(image_rgb)

                y_pred = model.predict([batch_inputs[0], batch_inputs[1], batch_inputs[2]])
                a = y_pred[0, 0:128]
                p = y_pred[0, 128:256]
                n = y_pred[0, 256:384]

                self.out_queue.put({'image_name': sample['a'], 'embedding': a})
                self.out_queue.put({'image_name': sample['p'], 'embedding': p})
                self.out_queue.put({'image_name': sample['n'], 'embedding': n})
                self.signal_queue.put(SENTINEL)

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
        manager = mp.Manager()
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
    # scan all files under img_path
    names = get_lfw_images()

    # init scheduler
    x = Scheduler(gpuids, q)

    # start processing and wait for complete
    return x.start(names)


SENTINEL = 1


def listener(q):
    pbar = tqdm(total=13233 // 3)
    for item in iter(q.get, None):
        pbar.update()


def inference():
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

    with open("data/preds.p", "wb") as file:
        pickle.dump(out_list, file)

    q.put(None)
    proc.join()


if __name__ == "__main__":
    if not os.path.isfile('data/preds.p'):
        inference()
    with open('data/preds.p', 'rb') as file:
        embeddings = pickle.load(file)

    pairs = get_pairs()
    y_true_list = []
    y_pred_list = []

    for pair in tqdm(pairs):
        image_name_1 = pair['image_name_1']
        image_name_2 = pair['image_name_2']
        y_true = pair['same_person']
        y_true_list.append(y_true)
        embedding_1 = np.array([x['embedding'] for x in embeddings if x['image_name'] == image_name_1][0])
        embedding_2 = np.array([x['embedding'] for x in embeddings if x['image_name'] == image_name_2][0])
        dist = np.square(np.linalg.norm(embedding_1 - embedding_2))
        y_pred = dist <= 0.9
        y_pred_list.append(y_pred)

    y = np.array(y_true_list).astype(np.int32)
    pred = np.array(y_pred_list).astype(np.int32)
    from sklearn import metrics

    print(y)
    print(pred)

    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print(metrics.auc(fpr, tpr))
