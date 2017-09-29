from abc import ABCMeta
from abc import abstractmethod
import re
import tensorflow as tf
import utils
from core import ops
import models.utils as mutils

slim = tf.contrib.slim


class EmbeddingModel(object, metaclass=ABCMeta):
    """Abstract base class for embedding model. """

    def __init__(self, fusion_layers, embedding_depth, feature_scope=None):
        self._fusion_layers = fusion_layers
        self._embedding_depth = embedding_depth
        self._feature_scope = feature_scope
        self._end_points = {}

    @abstractmethod
    def build(self, preprocessed_img, is_training=True, scope=None):
        """Build network and extract final embedding. """
        pass

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Preprocess for each backbone network. """
        pass

    def get_layer(self, name):
        """Get specified feature given name. """
        return self._end_points['{}/{}'.format(self._feature_scope, name)]

    def set_layer(self, name, val):
        self._end_points['{}/{}'.format(self._feature_scope, name)] = val

    def restore_fn(self, ckpt_path, from_embedding_checkpoint=False):
        """Initialize the network parameters from the pre-trained model.

        Args:
            ckpt_path: Path to the checkpoint.
            from_embedding_checkpoint: Whether restore from embedding's
                checkpoint.

        Returns:
            Function that takes a session and initializes the network.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            var_name = variable.op.name
            if not from_embedding_checkpoint:
                if var_name.startswith(self._feature_scope):
                    var_name = (re.split('^' + self._feature_scope + '/',
                                         var_name)[-1])
            variables_to_restore[var_name] = variable

        variables_to_restore = (utils.get_variables_available_in_checkpoint(
            variables_to_restore, ckpt_path))
        init_fn = slim.assign_from_checkpoint_fn(ckpt_path,
                                                 variables_to_restore)
        return init_fn

    def loss(self, loss_config, final_embedding, inst_label_batch):
        with tf.name_scope('loss'):
            embedding_losses = []
            for i, (kernel_size, strides, padding, dilation_rate) in enumerate(
                    zip(loss_config['kernel_size'], loss_config['strides'],
                        loss_config['padding'], loss_config['dilation_rate'])):
                embedding_pos_loss, embedding_neg_loss = ops.dense_siamese_loss(
                    final_embedding,
                    inst_label_batch,
                    kernel_size,
                    strides,
                    padding,
                    dilation_rate,
                    alpha=loss_config['alpha'],
                    beta=loss_config['beta'],
                    norm_ord=loss_config['norm_ord'],
                    normalize=loss_config['normalize'])
                embedding_loss = embedding_pos_loss + embedding_neg_loss
                embedding_losses.append(embedding_loss)
                utils.summary_scalar('embedding_pos_loss_{}'.format(i),
                                     embedding_pos_loss)
                utils.summary_scalar('embedding_neg_loss_{}'.format(i),
                                     embedding_neg_loss)
                utils.summary_scalar('embedding_loss_{}'.format(i),
                                     embedding_loss)
            embedding_loss = tf.add_n(embedding_losses)
            l2_loss = mutils.l2_loss(loss_config['weight_decay'])
            total_loss = embedding_loss + l2_loss
            utils.summary_scalar('l2_loss', l2_loss)
            utils.summary_scalar('embedding_loss', embedding_loss)
            utils.summary_scalar('total_loss', total_loss)
        return total_loss

    def _add_fusion_embedding(self, embedding_size, scope=None):
        """Combine several feature into final embedding. """

        def _initializer(shape,
                         seed=None,
                         dtype=tf.float32,
                         partition_info=None):
            return tf.truncated_normal(shape, 0.0, 1e-6, seed=seed, dtype=dtype)

        fusion_layers = [self.get_layer(key) for key in self._fusion_layers]
        with tf.variable_scope(scope, 'embedding', fusion_layers):
            fused = tf.concat(
                [
                    tf.image.resize_images(fmap, embedding_size)
                    for fmap in fusion_layers
                ],
                axis=-1)
            embedding = slim.conv2d(
                fused,
                self._embedding_depth, [1, 1],
                weights_initializer=_initializer,
                activation_fn=None,
                scope='embedding')
        self.set_layer('embedding', embedding)
        return embedding
