from __future__ import division
from abc import ABCMeta
from abc import abstractmethod
import re
import tensorflow as tf
import utils
from core import ops

slim = tf.contrib.slim


def weight_l2_loss(weight_decay, scope=None):
    """Compute l2 loss on weight.

    Args:
        weight_decay: weight of l2 loss.
        scope: tf scope.

    Returns:
        l2_losses: all l2 loss exclude bias term.
    """
    with tf.name_scope(scope, 'l2_loss'):
        l2_losses = tf.add_n([
            weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'weights' in v.name
        ])
    return l2_losses


def dense_siamese_loss(embeddings,
                       label,
                       kernel_size,
                       strides=(1, 1),
                       padding=(1, 1),
                       dilation_rate=(1, 1),
                       alpha=0.5,
                       beta=2.0,
                       norm_ord=1,
                       ignore_label=255,
                       normalize='none',
                       ignore_bg_pos=False,
                       data_balance=False,
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
        norm_ord: order of l-p norm.
        ignore_label: label to be ignored.
        normalize: method for embedding.
        ignore_bg_pos: whether or not ignore positive pair from background.
        data_balance: whether to balance neg/pos ratio.
        scope: name of scope.

    Returns:
        A scalar of total siamese loss.
    """
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

    w_size = kernel_size[0] * kernel_size[1]
    center_index = int((w_size - 1) / 2)
    with tf.name_scope(scope, 'dense_siamese_loss', [embeddings, label]):
        if normalize == 'l2':
            embeddings = tf.nn.l2_normalize(embeddings, 1)
        elif normalize == 'none':
            pass
        else:
            raise ValueError('Unknown normalize method: {}'.format(normalize))

        embedding_matrix = ops.im2col(embeddings, kernel_size, strides, padding,
                                      dilation_rate)
        distance_matrix = tf.norm(
            embedding_matrix -
            embedding_matrix[:, :, center_index:center_index + 1],
            ord=norm_ord,
            axis=1,
            keep_dims=True)
        label_matrix = ops.im2col(label, kernel_size, strides, padding,
                                  dilation_rate)

        label_center = label_matrix[:, :, center_index:center_index + 1]
        valid_mask = tf.logical_not(
            tf.logical_or(
                tf.equal(label_matrix, ignore_label),
                tf.equal(label_center, ignore_label)))
        mask = tf.equal(label_matrix, label_center)

        pos_mask = tf.logical_and(mask, valid_mask)
        if ignore_bg_pos:
            pos_mask = tf.logical_and(pos_mask, tf.not_equal(label_center, 0))
        pos_dist = tf.boolean_mask(distance_matrix, pos_mask)
        pos_loss = tf.reduce_sum(tf.maximum(0.0, pos_dist - alpha))
        neg_mask = tf.logical_and(tf.logical_not(mask), valid_mask)
        neg_dist = tf.boolean_mask(distance_matrix, neg_mask)
        neg_loss = tf.reduce_sum(tf.maximum(0.0, beta - neg_dist))

        num_pos = tf.count_nonzero(pos_mask)
        num_neg = tf.count_nonzero(neg_mask)
        normalizer = tf.to_float(num_pos + num_neg)
        if data_balance:
            neg_loss = neg_loss * tf.to_float(num_pos) / tf.to_float(num_neg)
        utils.summary_scalar('num/pos', num_pos)
        utils.summary_scalar('num/neg', num_neg)
        utils.summary_histogram('dist/neg', neg_dist)
        utils.summary_histogram('dist/pos', pos_dist)
    return pos_loss / normalizer, neg_loss / normalizer


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
                embedding_pos_loss, embedding_neg_loss = dense_siamese_loss(
                    final_embedding,
                    inst_label_batch,
                    kernel_size,
                    strides,
                    padding,
                    dilation_rate,
                    alpha=loss_config['alpha'],
                    beta=loss_config['beta'],
                    norm_ord=loss_config['norm_ord'],
                    normalize=loss_config['normalize'],
                    ignore_bg_pos=loss_config['ignore_bg_pos'],
                    data_balance=loss_config['data_balance'])
                embedding_loss = embedding_pos_loss + embedding_neg_loss
                embedding_losses.append(embedding_loss)
                utils.summary_scalar('embedding_pos_loss_{}'.format(i),
                                     embedding_pos_loss)
                utils.summary_scalar('embedding_neg_loss_{}'.format(i),
                                     embedding_neg_loss)
                utils.summary_scalar('embedding_loss_{}'.format(i),
                                     embedding_loss)
            embedding_loss = tf.add_n(embedding_losses)
            l2_loss = weight_l2_loss(loss_config['weight_decay'])
            total_loss = embedding_loss + l2_loss
            utils.summary_scalar('l2_loss', l2_loss)
            utils.summary_scalar('embedding_loss', embedding_loss)
            utils.summary_scalar('total_loss', total_loss)
        return total_loss

    def _add_fusion_embedding(self, embedding_size, scope=None, scale=5.0):
        """Combine several feature into final embedding. """

        def _initializer(shape,
                         seed=None,
                         dtype=tf.float32,
                         partition_info=None):
            return tf.truncated_normal(shape, 0.0, 1e-6, seed=seed, dtype=dtype)

        fusion_layers = []
        for key in self._fusion_layers:
            if 'input' in key:
                # approximately normalize to [0, scale]
                fusion_layers.append(
                    (self.get_layer(key) / 255.0 + 0.5) * scale)
            else:
                fusion_layers.append(self.get_layer(key))
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
