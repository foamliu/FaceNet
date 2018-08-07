# FaceNet

This is a Keras implementation of the face recognizer described in the paper [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

CelebFaces Attributes Dataset (CelebA) is a large-scale face dataset with 10,177 identities and 202,599 face images.

![image](https://github.com/foamliu/FaceNet/raw/master/images/CelebA.png)

Follow the [instruction](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to download Large-scale CelebFaces Attributes (CelebA) Dataset.

## Architecture
![image](https://github.com/foamliu/FaceNet/raw/master/images/model.png)

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo

Download [pre-trained model](https://github.com/foamliu/Look-Into-Person/releases/download/v1.0/model.119-2.2473.hdf5) and put it into models folder.

```bash
$ python demo.py
```

P | d | A | d | N |
|---|---|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|$(distance_0_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|$(distance_0_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|$(distance_1_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|$(distance_1_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|$(distance_2_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|$(distance_2_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|$(distance_3_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|$(distance_3_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|$(distance_4_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|$(distance_4_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|$(distance_5_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|$(distance_5_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|$(distance_6_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|$(distance_6_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|$(distance_7_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|$(distance_7_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|$(distance_8_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|$(distance_8_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|$(distance_9_a_p)|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|$(distance_9_a_n)|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|