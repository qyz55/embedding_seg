import tensorflow as tf
import utils
from core import ops

slim = tf.contrib.slim


def vgg_preprocess_img(resized_inputs):
    """Preprocess fn for vgg net. """
    with tf.name_scope('preprocess'):
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        return tf.cast(
            tf.cast(resized_inputs, dtype=tf.float32) -
            [_R_MEAN, _G_MEAN, _B_MEAN], tf.float32)


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
        weight_decay: The l2 regularization coefficient.

    Returns:
        An arg_scope.
    """
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_embedding(inputs, scope=None):
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        h, w = ops.get_shape_list(inputs)[1:3]
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            conv1 = slim.repeat(
                inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            conv1_pool = slim.max_pool2d(conv1, [2, 2], scope='pool1')
            conv2 = slim.repeat(
                conv1_pool, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            conv2_pool = slim.max_pool2d(conv2, [2, 2], scope='pool2')
            conv3 = slim.repeat(
                conv2_pool, 3, slim.conv2d, 256, [3, 3], scope='conv3')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)
        return conv3, end_points


def add_fusion_embedding(end_points,
                         fusion_layers,
                         embedding_size,
                         embedding_depth=64,
                         scope=None):

    def _initializer(shape, seed=None, dtype=tf.float32, partition_info=None):
        """Initializer for locization's weight.

        Ensure the initial behavior of locization refinement is nearly
        trivial.
        """
        return tf.truncated_normal(shape, 0.0, 1e-6, seed=seed, dtype=dtype)

    fusion_layers = [end_points[key] for key in fusion_layers]
    with tf.variable_scope(scope, 'embedding', fusion_layers):
        fused = tf.concat(
            [
                tf.image.resize_images(fmap, embedding_size)
                for fmap in fusion_layers
            ],
            axis=-1)
        embedding = slim.conv2d(
            fused,
            embedding_depth, [1, 1],
            weights_initializer=_initializer,
            activation_fn=None,
            scope='embedding')
    end_points.update({'embedding': embedding})
    return embedding


def build_model(inputs, fusion_layers, weight_decay=5e-4):
    with slim.arg_scope(vgg_arg_scope(weight_decay=weight_decay)):
        net, end_points = vgg_embedding(inputs)
        end_points['input'] = inputs
        embedding = add_fusion_embedding(end_points, fusion_layers,
                                         tf.shape(inputs)[1:3], 64)
    for key, val in end_points.items():
        utils.summary_histogram('output/{}'.format(key), val)
    return embedding, end_points


def load_vgg_imagenet(ckpt_path, scope_name=None):
    """Initialize the network parameters from the VGG-16 pre-trained model.

    Args:
        Path to the checkpoint

    Returns:
        Function that takes a session and initializes the network
    """
    variables_to_restore = {}
    all_variables = tf.global_variables()
    for variable in all_variables:
        var_name = variable.op.name
        if scope_name is not None:
            var_name.replace("vgg_16", scope_name)
        variables_to_restore[var_name] = variable
    variables_to_restore = (utils.get_variables_available_in_checkpoint(
        variables_to_restore, ckpt_path))
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, variables_to_restore)
    return init_fn
