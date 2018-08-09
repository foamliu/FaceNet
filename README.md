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

P | distance | A | distance | N |
|---|---|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.4480|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|1.7922|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|2.2238|---|2.5129|---|2.2644|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.1176|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|1.0159|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|2.2250|---|2.2095|---|0.5814|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.0727|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.6875|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|1.9619|---|2.2471|---|1.2657|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.2634|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|1.4948|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|2.0321|---|1.8956|---|2.5378|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.7871|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|0.9358|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|1.7645|---|0.7248|---|1.1630|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.1003|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|0.9647|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|2.6672|---|2.3015|---|0.7265|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.6239|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|1.8791|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|2.0709|---|1.9866|---|2.1705|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.3224|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|2.0867|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|2.2542|---|1.9866|---|2.6825|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.9798|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|2.2034|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|2.5072|---|1.8167|---|0.8505|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|0.1785|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|1.0298|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|

### Evaluation

Labeled Faces in the Wild (LFW) database info:

- 13233 images
- 5749 people
- 1680 people with two or more images

Download the [LFW database](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz) and put it under data folder:

```bash
$ wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
$ tar -xvf lfw-funneled.tgz
$ wget http://vis-www.cs.umass.edu/lfw/pairs.txt
$ wget http://vis-www.cs.umass.edu/lfw/people.txt

$ python lfw_eval.py
```