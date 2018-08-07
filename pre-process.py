import os
import zipfile

import cv2 as cv
import dlib
from tqdm import tqdm

from config import identity_annot_filename, image_folder, img_size
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def check_image():
    detector = dlib.get_frontal_face_detector()
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()

    for line in tqdm(lines):
        line = line.strip()
        if len(line) > 0:
            tokens = line.split(' ')
            image_name = tokens[0].strip()
            id = tokens[1].strip()
            filename = os.path.join(image_folder, image_name)
            try:
                image_bgr = cv.imread(filename)
                image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
                dets = detector(image_rgb, 1)
                if len(dets) > 0:
                    d = dets[0]
                    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
                    image_rgb = image_rgb[y1:y2 + 1, x1:x2 + 1]
                image_rgb = cv.resize(image_rgb, (img_size, img_size), cv.INTER_CUBIC)
            except cv.error as err:
                print(err)
                print(filename)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    # if not os.path.isdir(image_folder):
    extract(image_folder)

    check_image()
