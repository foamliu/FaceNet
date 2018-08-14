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
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.6187|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|1.5339|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|1.0369|---|1.2335|---|0.8919|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.3095|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|0.5787|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|1.3969|---|1.1069|---|0.6919|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.3180|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.3270|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|1.8087|---|1.4229|---|1.1394|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.2450|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|1.3762|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|1.1232|---|1.4875|---|2.7506|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.5540|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|1.2083|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|0.3829|---|0.5497|---|1.8303|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.4328|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|1.0244|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|1.2141|---|0.8059|---|1.3821|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.6969|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|0.7049|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|0.8955|---|1.0979|---|1.0045|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.8061|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|0.9656|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|1.1230|---|1.5219|---|1.2661|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.5752|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|1.3030|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|1.4187|---|1.0789|---|1.5365|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|1.3583|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|0.9191|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|

### Evaluation

Labeled Faces in the Wild (LFW) database info:

- 13233 images
- 5749 people
- 1680 people with two or more images

Download the [LFW database](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) and put it under data folder:

```bash
$ wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
$ tar -xvf lfw-funneled.tgz
$ wget http://vis-www.cs.umass.edu/lfw/pairs.txt
$ wget http://vis-www.cs.umass.edu/lfw/people.txt

$ python lfw_eval.py
```

Accuracy is: 80.57 %.