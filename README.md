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
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.3372|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|2.0129|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|1.3138|---|1.8652|---|1.7139|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.5115|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|1.6192|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|1.2628|---|1.6822|---|1.5982|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.7592|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.8448|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|1.1273|---|1.6227|---|2.0344|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.2926|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|1.2980|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|1.0961|---|1.1603|---|1.2846|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.2803|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|1.1043|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|1.3909|---|1.6874|---|1.6249|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.1778|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|1.7960|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|0.8747|---|0.9101|---|1.4294|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|1.1066|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|1.4510|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|1.8802|---|1.6793|---|2.0304|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.5532|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|1.8857|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|1.9119|---|1.9540|---|2.9826|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.7046|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|0.5064|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|2.3759|---|2.2039|---|0.7147|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|0.3689|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|1.5293|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|

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