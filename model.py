import keras.backend as K
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras.utils import plot_model

from config import img_size, channel, embedding_size


def build_model():
    # image embedding
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, channel),
                                   pooling='avg')
    image_input = base_model.input
    x = base_model.layers[-1].output
    x = Dense(embedding_size)(x)
    output = x
    model = Model(inputs=image_input, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
