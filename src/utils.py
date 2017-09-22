import sys
import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

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
            print('Variable [%s] not available in checkpoint', variable_name, file=sys.stderr)
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

def summary_embedding(sess, name, tensor, method="pca"):
    """Visualization using dimension reduction method.
    Args:
        sess: Session used.
        name: name in tensorboard.
        tensor: [b, h, w, c] tensor to be visualized.
        method: method used for dimension reduction.
    """
    mask = sess.run(tensor)
    b, h, w, c = mask.shape
    output = np.zeros((b, h, w, 3), dtype = uint8)
    for i in range b:
        tmp = np.zeros((h*w, c), dtype = float)
        for j_, j in enumerate(mask[i, :, :, :]):
            for k_, k in enumerate(j):
                for l_, l in enumerate(k):
                    tmp[j_*w + k_, l_] = l
        pca = PCA(n_components=3)
        re = pca.fit_transform(tmp)
        re = re.reshape(h,w,3)
        re = np.uint8((re-re.min())/(re.max()-re.min())*255)
        output[i] = np.array(re)
        #img = Image.fromarray(output[i])
        #img.save('./mask'+str(i)+'.png')
    output_tensor = tf.convert_to_tensor(output, dtype = tf.uint8, name = name)
    tf.summary.image(name, output_tensor,max_outputs = b ,collections=['detailed',tf.GraphKeys.SUMMARIES])
