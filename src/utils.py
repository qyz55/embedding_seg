from __future__ import division

import sys
import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np


def get_variables_available_in_checkpoint(variables, checkpoint_path):
    """Returns the subset of variables available in the checkpoint.

    Inspects given checkpoint and returns the subset of variables that are
    available in it.

    TODO: force input and output to be a dictionary.

    Args:
        variables: a list or dictionary of variables to find in checkpoint.
        checkpoint_path: path to the checkpoint to restore variables from.

    Returns:
        A list or dictionary of variables.
    Raises:
        ValueError: if `variables` is not a list or dict.
    """
    if isinstance(variables, list):
        variable_names_map = {
            variable.op.name: variable
            for variable in variables
        }
    elif isinstance(variables, dict):
        variable_names_map = variables
    else:
        raise ValueError('`variables` is expected to be a list or dict.')
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars = ckpt_reader.get_variable_to_shape_map().keys()
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variable_names_map.items()):
        if variable_name in ckpt_vars:
            vars_in_ckpt[variable_name] = variable
        else:
            print(
                'Variable [%s] not available in checkpoint',
                variable_name,
                file=sys.stderr)
    if isinstance(variables, list):
        return vars_in_ckpt.values()
    return vars_in_ckpt


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


def summary_embedding(name, tensor, num_save_images=2, method="pca"):
    """Visualization using dimension reduction method.
    Args:
        name: name in tensorboard.
        tensor: [b, h, w, c] tensor to be visualized.
        num_save_images: number of images to be summaried.
        method: method used for dimension reduction.
    """
    if method == "pca":
        result = tf.py_func(pca_np, [tensor, num_save_images], tensor.dtype)
    else:
        raise ValueError(
            "Unknown dimension reduction method: {}".format(method))
    tmin, tmax = tf.reduce_min(tensor), tf.reduce_max(tensor)
    result = tf.cast((result - tmin) / (tmax - tmin) * 255.0, tf.uint8)

    tf.summary.image(
        name,
        result,
        max_outputs=num_save_images,
        #  collections=['detailed', tf.GraphKeys.SUMMARIES])  # TODO(meijieru)
        collections=['brief', 'detailed', tf.GraphKeys.SUMMARIES])


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
