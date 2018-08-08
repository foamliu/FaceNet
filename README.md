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
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.1119|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|2.2067|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|2.5508|---|2.1330|---|2.1488|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.2414|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|2.1896|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|1.9324|---|1.1936|---|0.9306|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.3651|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|0.7959|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|0.6656|---|1.1787|---|2.0192|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.1274|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|2.0338|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|1.2064|---|0.3345|---|2.1961|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|1.1199|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|0.9894|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|1.3017|---|0.4430|---|0.8627|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.3806|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|0.1114|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|0.6295|---|0.5923|---|1.4121|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.2571|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|1.5465|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|0.4731|---|0.8136|---|1.5707|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.1284|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|0.7770|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|0.5258|---|0.6880|---|0.1182|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.1443|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|0.2833|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|0.3525|---|0.3856|---|0.4070|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|0.3409|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|0.2622|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|