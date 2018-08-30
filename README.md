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

Download [pre-trained model](https://github.com/foamliu/FaceNet/releases/download/v1.0/model.10-0.0156.hdf5) and put it into models folder.

```bash
$ python demo.py
```

P | distance | A | distance | N |
|---|---|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.1716|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|1.6495|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|1.2839|---|1.1502|---|1.1636|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.3566|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|0.9795|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|1.6029|---|1.5733|---|1.2582|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.7500|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.2708|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|1.4815|---|1.0065|---|1.7432|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.2974|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|1.2198|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|2.0759|---|1.6838|---|1.3330|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.3072|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|1.2609|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|0.5769|---|0.7416|---|0.8989|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.3422|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|0.4381|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|1.4096|---|1.7690|---|1.0634|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.5896|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|1.3287|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|1.7525|---|1.5093|---|0.9600|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.5894|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|1.4106|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|1.5781|---|0.7706|---|1.7681|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.6818|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|0.8294|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|1.1007|---|0.8181|---|1.1559|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|0.3873|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|0.9675|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|

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
