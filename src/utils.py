import tensorflow as tf


def summary_scalar(name, scalar):
    tf.summary.scalar(
        name, scalar, collections=['brief', 'detailed', tf.GraphKeys.SUMMARIES])


def summary_histogram(name, tensor_or_list):
    if isinstance(tensor_or_list, tf.Tensor) or isinstance(
            tensor_or_list, tf.Variable):
        tensor = tensor_or_list
    elif isinstance(tensor_or_list, list):
        tensor = tf.concat(
            [tf.reshape(v, [-1]) for v in tensor_or_list], axis=0)
    else:
        raise ValueError('Unsupported type: {}'.format(type(tensor_or_list)))

    tf.summary.histogram(
        name, tensor, collections=['detailed', tf.GraphKeys.SUMMARIES])
