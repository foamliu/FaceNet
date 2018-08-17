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
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.3579|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|0.9037|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|1.2757|---|1.0052|---|1.0740|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.7775|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|0.7634|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|1.7359|---|1.3784|---|0.7606|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.3556|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.3154|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|1.1428|---|0.9498|---|0.7804|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.5836|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|1.8520|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|2.0298|---|1.9031|---|1.9189|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.3343|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|1.7794|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|1.3242|---|1.5383|---|0.6461|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.2490|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|1.0756|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|1.2015|---|1.4692|---|0.8207|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.3323|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|0.9058|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|1.1681|---|1.1821|---|0.5887|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.6769|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|0.8985|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|1.1974|---|0.7921|---|1.3920|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.3638|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|1.4623|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|0.8634|---|0.8968|---|0.5992|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|0.3681|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|1.3735|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|

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

Accuracy is: 89.27 %.

### Image Augmentation
before | after | before | after | before | after |
|---|---|---|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_0.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_0.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_1.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_1.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_2.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_2.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_3.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_3.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_4.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_4.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_5.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_5.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_6.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_6.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_7.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_7.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_8.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_8.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_9.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_9.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_10.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_10.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_11.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_11.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_12.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_12.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_13.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_13.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_14.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_14.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_15.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_15.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_16.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_16.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_17.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_17.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_18.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_18.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_19.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_19.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_20.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_20.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_21.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_21.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_22.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_22.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_23.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_23.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_24.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_24.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_25.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_25.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_26.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_26.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_27.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_27.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_28.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_28.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_29.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_29.png)|
