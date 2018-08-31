import bz2
import os
import zipfile
from multiprocessing import Pool

import cv2 as cv
import dlib
from tqdm import tqdm

from config import img_size, identity_annot_filename, image_folder
from utils import ensure_folder

predictor_path = 'models/shape_predictor_5_face_landmarks.dat'


def ensure_dlib_model():
    if not os.path.isfile(predictor_path):
        import urllib.request
        urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
                                   filename="models/shape_predictor_5_face_landmarks.dat.bz2")


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def extract_bz2(new):
    old = '{}.bz2'.format(new)
    print('Extracting {}...'.format(old))
    with open(new, 'wb') as new_file, bz2.BZ2File(old, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)


def check_one_image(line):
    line = line.strip()
    if len(line) > 0:
        tokens = line.split(' ')
        image_name = tokens[0].strip()
        # print(image_name)
        filename = os.path.join(image_folder, image_name)
        # print(filename)
        img = cv.imread(filename)
        img = img[:, :, ::-1]
        dets = detector(img, 1)

        num_faces = len(dets)
        if num_faces == 0:
            return image_name

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        # It is also possible to get a single chip
        image = dlib.get_face_chip(img, faces[0], size=img_size)
        image = image[:, :, ::-1]


        # try:
        #     resized = cv.resize(original, (img_size, img_size), cv.INTER_CUBIC)
        # except cv.error as err:
        #     print(filename)
        #     print('image_name={} original.shape={}'.format(image_name, original.shape))
        #     print('image_name={} resized.shape={}'.format(image_name, resized.shape))


def check_image():
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()
    # check_one_image(lines[0])

    pool = Pool(24)
    results = []
    for item in tqdm(pool.imap_unordered(check_one_image, lines), total=len(lines)):
        results.append(item)
    pool.close()
    pool.join()

    results = [r for r in results if r is not None]
    print(len(results))
    with open('results.txt', 'w') as file:
        file.write('\n'.join(results))


if __name__ == '__main__':
    ensure_folder('data')
    ensure_folder('models')
    ensure_dlib_model()
    extract_bz2(predictor_path)

    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    if not os.path.isdir(image_folder):
        extract(image_folder)

    check_image()
