import tensorflow as tf


def vgg_preprocess_img(resized_inputs):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    return tf.cast(
        tf.cast(resized_inputs, dtype=tf.float32) - [_R_MEAN, _G_MEAN, _B_MEAN],
        tf.float32)


def l2_loss(weight_decay, scope=None):
    """Compute l2 loss on weight.

    Args:
        weight_decay: weight of l2 loss.
        scope: tf scope.

    Returns:
        l2_losses: all l2 loss exclude bias term.
    """
    with tf.name_scope(scope, 'l2_loss'):
        l2_losses = tf.add_n([
            weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'weights' in v.name
        ])
    return l2_losses
