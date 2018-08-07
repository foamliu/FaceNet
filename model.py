import keras.backend as K
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras.utils import plot_model

from config import img_size, channel, embedding_size


def build_model():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, channel),
                                   pooling='avg')
    image_input = base_model.input
    x = base_model.layers[-1].output
    out = Dense(embedding_size)(x)
    image_embedder = Model(image_input, out)

    input_a = Input((img_size, img_size, channel), name='anchor')
    input_p = Input((img_size, img_size, channel), name='positive')
    input_n = Input((img_size, img_size, channel), name='negative')

    normalize = Lambda(lambda x: x / tf.norm(x), name='normalize')

    x = image_embedder(input_a)
    output_a = normalize(x)
    x = image_embedder(input_p)
    output_p = normalize(x)
    x = image_embedder(input_n)
    output_n = normalize(x)

    merged_vector = concatenate([output_a, output_p, output_n], axis=-1)

    model = Model(inputs=[input_a, input_p, input_n],
                  outputs=merged_vector)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
