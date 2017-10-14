import tensorflow as tf
import utils
from core import ops
from models import meta
from models import common

slim = tf.contrib.slim


def vgg_arg_scope():
    """Defines the VGG arg scope.

    Args:
        weight_decay: The l2 regularization coefficient.

    Returns:
        An arg_scope.
    """
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_embedding(inputs, build_full_model=False, scope=None):
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

            if build_full_model:
                #  TODO(meijieru)
                raise NotImplementedError()

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)
        return conv3, end_points


class VggEmbeddingModel(meta.EmbeddingModel):
    """VGG embedding model. """

    def __init__(self, *args, **kwargs):
        super(VggEmbeddingModel, self).__init__(
            *args, feature_scope='vgg', **kwargs)
        if self._seg_branch_config['output_stride'] != 32:
            raise ValueError('Vgg does not support adjustable `output_stride`')

    def _extract_feature(self, preprocessed_img, is_training=True, scope=None):
        """Build base network. """
        self.set_layer('input', preprocessed_img)
        with tf.variable_scope(self._feature_scope, 'embedding',
                               [preprocessed_img]):
            with slim.arg_scope(vgg_arg_scope()):
                net, end_points = vgg_embedding(
                    preprocessed_img,
                    build_full_model=self._seg_branch_config['use'])
                self._end_points.update(end_points)
        return net

    def preprocess(self, resized_inputs):
        """Preprocess fn for vgg net. """
        with tf.name_scope('preprocess'):
            return common.vgg_preprocess_img(resized_inputs)
