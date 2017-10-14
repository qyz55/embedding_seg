import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
import utils
from models import meta
from models import common

slim = tf.contrib.slim


def resnet_v1_50_embedding(inputs,
                           num_block=4,
                           num_classes=None,
                           is_training=None,
                           global_pool=True,
                           output_stride=None,
                           reuse=None,
                           scope='resnet_v1_50'):
    blocks = [
        resnet_v1.resnet_v1_block(
            'block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block(
            'block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block(
            'block3', base_depth=256, num_units=6, stride=2),
        resnet_v1.resnet_v1_block(
            'block4', base_depth=512, num_units=3, stride=1),
    ]
    blocks = blocks[:num_block]
    return resnet_v1.resnet_v1(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)


def resnet_v1_101_embedding(inputs,
                            num_block=4,
                            num_classes=None,
                            is_training=None,
                            global_pool=True,
                            output_stride=None,
                            reuse=None,
                            scope='resnet_v1_101'):
    blocks = [
        resnet_v1.resnet_v1_block(
            'block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block(
            'block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_v1.resnet_v1_block(
            'block4', base_depth=512, num_units=3, stride=1),
    ]
    blocks = blocks[:num_block]
    return resnet_v1.resnet_v1(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)


class ResnetEmbeddingModel(meta.EmbeddingModel):
    """Resnet embedding base class. """

    def __init__(self, *args, **kwargs):
        self._architecture = kwargs.pop('architecture')
        super(ResnetEmbeddingModel, self).__init__(*args, **kwargs)

    def _extract_feature(self, preprocessed_img, is_training=True, scope=None):
        """Build base network. """
        self.set_layer('input', preprocessed_img)
        with tf.variable_scope(self._feature_scope, 'embedding',
                               [preprocessed_img]):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = self._architecture(
                    preprocessed_img,
                    num_block=4 if self._seg_branch_config['use'] else 3,
                    is_training=is_training,
                    output_stride=self._seg_branch_config['output_stride'],
                    num_classes=None,
                    global_pool=False)
                self._end_points.update(end_points)
        return net

    def preprocess(self, resized_inputs):
        """Preprocess fn for resnet. """
        with tf.name_scope('preprocess'):
            return common.vgg_preprocess_img(resized_inputs)


class Resnet50EmbeddingModel(ResnetEmbeddingModel):
    """Resnet-50 embedding model. """

    def __init__(self, *args, **kwargs):
        super(Resnet50EmbeddingModel, self).__init__(
            *args,
            feature_scope='resnet50',
            architecture=resnet_v1_50_embedding,
            **kwargs)


class Resnet101EmbeddingModel(ResnetEmbeddingModel):
    """Resnet-101 embedding model. """

    def __init__(self, *args, **kwargs):
        super(Resnet101EmbeddingModel, self).__init__(
            *args,
            feature_scope='resnet101',
            architecture=resnet_v1_101_embedding,
            **kwargs)
