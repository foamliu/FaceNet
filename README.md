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
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.9787|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|1.3693|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|1.9202|---|1.9520|---|2.0004|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.7962|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|1.4382|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|2.4617|---|2.7121|---|2.4466|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.4633|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.5729|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|2.4821|---|2.5138|---|1.5836|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.5206|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|2.1650|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|1.0211|---|0.9185|---|1.3342|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.5970|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|1.5701|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|1.6693|---|1.0589|---|1.0968|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.4030|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|1.5000|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|2.1327|---|1.7920|---|1.6252|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.4124|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|1.6971|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|1.8121|---|1.4903|---|1.3148|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.3632|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|0.9825|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|2.1700|---|2.1616|---|2.2323|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.6451|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|1.2369|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|1.9688|---|0.9834|---|1.4636|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|1.2756|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|1.5089|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|

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

Accuracy is: 79.52 %.