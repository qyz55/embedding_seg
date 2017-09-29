import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
import utils
from models import meta
import models.utils as mutils

slim = tf.contrib.slim


def resnet_v1_50_embedding(inputs,
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
    ]
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
    ]
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

    def build(self, preprocessed_img, is_training=True, scope=None):
        """Build network and extract final embedding. """
        self.set_layer('input', preprocessed_img)
        with tf.variable_scope(self._feature_scope, 'embedding',
                               [preprocessed_img]):

            # inputs has shape [batch, 513, 513, 3]
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = self._architecture(
                    preprocessed_img,
                    is_training=is_training,
                    global_pool=False)
                self._end_points.update(end_points)
                embedding = self._add_fusion_embedding(
                    tf.shape(preprocessed_img)[1:3])
        for key, val in self._end_points.items():
            utils.summary_histogram('output/{}'.format(key), val)
        return embedding

    def preprocess(self, resized_inputs):
        """Preprocess fn for resnet. """
        with tf.name_scope('preprocess'):
            return mutils.vgg_preprocess_img(resized_inputs)


class Resnet50EmbeddingModel(ResnetEmbeddingModel):
    """Resnet-50 embedding model. """

    def __init__(self, *args, **kwargs):
        super(Resnet101EmbeddingModel, self).__init__(
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
