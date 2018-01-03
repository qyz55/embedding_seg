from __future__ import division

import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import utils

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


def tf_print(t, first_n=None, prefix=None, show_val=False, summarize=10):
    """Wrapper for tensorflow print. """
    with tf.name_scope('print'):
        to_print = [
            tf.reduce_mean(t),
            tf.reduce_min(t),
            tf.reduce_max(t),
            tf.shape(t)
        ]
        if show_val:
            to_print.append(t)
        return tf.Print(
            t, to_print, message=prefix, first_n=first_n, summarize=summarize)


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
           merge_spatial=True,
           data_format="NHWC"):
    """Convert image into matrix.

    Args:
        inputs: [b, h, w, c] tensor.
        kernel_size: (kh, kw) tuple.
        strides: (sh, sw) tuple.
        padding: how to pad the feature map.
        dilation_rate: (dh, dw) tuple.
        merge_spatial: whether to merge `oh`, `ow` dim.
        data_format: ["NCHW", "NHWC"]

    Returns:
        A [b, c, w_size, oh * ow] tensor if merge_spatial, otherwise a [b, c,
            wsize, oh, ow] tensor.
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
    if merge_spatial:
        result = dim_merge(result, [0, 1, 2, [3, 4]])  # [b, c, w_size, oh * ow]
    return result


def embedding(tensor, num_save_images=2, method="pca", epsilon=1e-8):
    """Visualization using dimension reduction method.

    Args:
        tensor: [b, h, w, c] tensor to be visualized.
        num_save_images: number of images to be summaried.
        method: method used for dimension reduction.
        epsilon: avoid ZeroDivisionError.

    Returns:
        A [n, h, w, 3] tensor within range [0, 255].
    """
    assert get_shape_list(tensor)[-1] > 3
    if method == "pca":
        result = tf.py_func(
            pca_np, [tensor, num_save_images], tensor.dtype, name='embedding')
    else:
        raise ValueError(
            "Unknown dimension reduction method: {}".format(method))
    
    utils.summary_histogram('pca', tensor[..., 0])
    '''
    with tf.control_dependencies([tf_print(tensor[0,150:160, 150:160, 0])]):
        tensor = tf_print(tensor)
    '''
    tmin, tmax = tf.reduce_min(tensor), tf.reduce_max(tensor)
    result = tf.cast((result - tmin) / (tmax - tmin + epsilon) * 255.0,
                     tf.uint8)
    return result


def pca_np(array, num_images, reduced_dim=3):
    b, h, w, c = array.shape
    assert b >= num_images
    assert c >= 3

    output = np.zeros((num_images, h, w, reduced_dim), dtype=array.dtype)
    for i in range(num_images):
        pca = PCA(n_components=3)
        result = pca.fit_transform(array[i].reshape([-1, c])).reshape([h, w, 3])
        output[i] = np.array(result)
    return output

def pick_k(inputs, k, max_instance, ignore_label = 255):
    b, h, w, c = inputs.shape
    result = np.zeros([b, max_instance, k], dtype = np.int32)
    num = np.zeros([b], dtype = np.int32)
    size = np.zeros([b, max_instance], dtype = np.int32)
    for s in range(b):
        g = inputs[s].reshape(-1)
        num_inst = np.unique(g)
        num_inst = [x for x in num_inst if x != ignore_label]
        num_tmp = len(num_inst)
        j = 0
        for i in range(num_tmp):
            col = np.where(g == num_inst[i])
            if (len(col[0]) >= k):
                ck = np.random.choice(col[0], k, False)
                result[s, j, :] = ck[:]
                size[s, j] = len(col[0])
                j = j + 1
        num[s] = j
    return result, num, size



def random_pick_k(inputs, k, max_instance = 20):
    """Pick k points from each instace including background.
    
    Args:
        inpus: [b, h, w, 1] tensor of image mask.
        k: points to be picked from each instnce.
        max_instance: the max instances in an image.

    Returns:
        A [b, max_instance, k] tensor, blank part(instance order
        that exceeds current number of instances) filled with -1.
        A [b] tensor, each element represents the number of instances in a picture.
        A [b, max_instance] tensor, the size of each instance.
    """
    b, h, w, c = inputs.shape
    [b, h, w, c] = [b.value, h.value, w.value, c.value]
    assert c == 1

    re, nu, si = tf.py_func(pick_k, [inputs, k, max_instance], [tf.int32, tf.int32, tf.int32], stateful = True, name = 'pick_k')

    '''
        arr = tf.range(0,max_instance + 1, 1)
        result = tf.ones([b, max_instance, k, 2], dtype = tf.int32) * (-1)
        for i in range(b):
            g = tf.reshape(inputs[i],[-1])
            num_inst, _ = tf.unique(g)
            print(tf.shape(num_inst).value)
            print(num_inst.shape)
            length = num_inst.get_shape()[0].value
            for j in range(length):
                col = tf.reshape(tf.where(tf.equal(g,arr[j])),-1)
                if (col.shape[0].value >= k):
                    ck, _, _ = tf.nn.uniform_candidate_sampler(tf.expand_dims(col, axis = 0), col.shape[0].value, k, True, col.shape[0].value)
                    results[i, j, :, 0] = ck[:] // w
                    results[i, j, :, 1] = ck[:] % w
    '''
    return re, nu, si


