import tensorflow as tf


def vgg_preprocess_img(resized_inputs):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    return tf.cast(
        tf.cast(resized_inputs, dtype=tf.float32) - [_R_MEAN, _G_MEAN, _B_MEAN],
        tf.float32)
