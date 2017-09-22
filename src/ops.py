import os
import numpy as np
import tensorflow as tf

libembedding_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'libembedding.so')
libembedding = tf.load_op_library(libembedding_path)


def get_shape_list(tensor):
    return tensor.get_shape().as_list()


def get_shape(tensor):
    """Get tensor's shape.

    If the size of a special dimension could be inferred by tensorflow, it will
    be a int, otherwise dynamic size of that dimension will be used.
    """
    static_shape = get_shape_list(tensor)
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [
        s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)
    ]
    return dims


def dim_merge(tensor, dims_list):
    """Reshape by specified dimension config.

    Args:
        dims_list: A list of mixed `int` or list,
    """
    shape = get_shape(tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, int):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(tf.reduce_prod([shape[d] for d in dims]))
    tensor = tf.reshape(tensor, dims_prod)
    return tensor


@tf.RegisterGradient("Im2Col")
def _im2col_grad(op, grad):
    kernel_size = op.get_attr("kernel_size")
    dilation_rate = op.get_attr("dilation_rate")
    padding = op.get_attr("padding")
    strides = op.get_attr("strides")
    data_format = op.get_attr("data_format")
    input_size = tf.shape(op.inputs[0])
    return libembedding.im2_col_grad(
        input_size,
        grad,
        kernel_size,
        strides,
        padding,
        dilation_rate,
        data_format=data_format)


def im2col(inputs,
           kernel_size,
           strides,
           padding,
           dilation_rate,
           data_format="NHWC"):
    """Convert image into matrix.

    Args:
        inputs: [b, h, w, c] tensor.
        kernel_size: (kh, kw) tuple.
        strides: (sh, sw) tuple.
        padding: how to pad the feature map.
        dilation_rate: (dh, dw) tuple.
        data_format: ["NCHW", "NHWC"]

    Returns:
        A [b, c, w_size, oh * ow] tensor.
    """
    if data_format == "NHWC":
        inputs = tf.transpose(inputs, [0, 3, 1, 2])  # NCHW
    result = libembedding.im2_col(
        inputs,
        kernel_size,
        strides,
        padding,
        dilation_rate,
        data_format="NCHW")  # [b, c, w_size, oh, ow]
    result = dim_merge(result, [0, 1, 2, [3, 4]])  # [b, c, w_size, oh * ow]
    return result


def dense_siamese_loss(embeddings,
                       label,
                       kernel_size,
                       strides=(1, 1),
                       padding=(1, 1),
                       dilation_rate=(1, 1),
                       alpha=0.5,
                       beta=2.0,
                       norm_ord=1,
                       scope=None):
    """Siamese loss within image.

    Encourage positions within the same instance has similar embeddings, while
    positions from different instance has large distance.

    Args:
        embeddings: [b, h, w, c] tensor, embeddings of input images.
        label: [b, h, w, 1] tensor, instance label of corresponding images.
        kernel_size: (kh, kw) tuple.
        strides: (sh, sw) tuple.
        padding: how to pad the feature map.
        dilation_rate: (dh, dw) tuple.
        alpha: threshold for positive pairs.
        beta: threshold for negative pairs.

    Returns:
        A scalar of total siamese loss.
    """
    w_size = kernel_size[0] * kernel_size[1]
    assert w_size % 2 == 1
    center_index = int((w_size - 1) / 2)

    with tf.name_scope(scope, 'dense_siamese_loss', [embeddings, label]):
        embedding_matrix = im2col(embeddings, kernel_size, strides, padding,
                                  dilation_rate)
        label_matrix = im2col(label, kernel_size, strides, padding,
                              dilation_rate)
        mask = tf.cast(
            (label_matrix == label_matrix[:, :, center_index:center_index + 1]),
            tf.float32)
        distance_matrix = tf.norm(
            embedding_matrix -
            embedding_matrix[:, :, center_index:center_index + 1],
            ord=norm_ord,
            axis=-1,
            keep_dims=True)

        #  FIXME(meijieru): sum or mean?
        pos_loss = tf.reduce_mean(
            mask * tf.maximum(0.0, distance_matrix - alpha))
        neg_loss = tf.reduce_mean(
            (1 - mask) * tf.maximum(0.0, beta - distance_matrix))
    return pos_loss, neg_loss
