from __future__ import division

import os
import sys
import shutil
import tensorflow as tf


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


def copy_to(src, dst):
    """Copy file from src to dst.

    Args:
        src: path for src file.
        dst: path for dst directory
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    shutil.copy(src, dst)


def construct_dataset(base_dir,
                      content,
                      save_path,
                      num_reader=1,
                      ignore_label=255,
                      input_size=(161, 161)):
    with open(save_path, 'w') as f:
        f.write('\n'.join(content))
    input_config = {
        "data_dir": base_dir,
        "data_list": save_path,
        "num_reader": num_reader,
        "ignore_label": ignore_label,
        "input_size": input_size
    }
    return input_config
