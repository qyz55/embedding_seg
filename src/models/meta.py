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


def multi_scale_dense_siamese_loss(loss_config,
                                   predict_dict,
                                   gt_dict,
                                   scope=None):
    """Multi-scale dense siamese loss. """
    if not loss_config['use']:
        return 0.0

    embeddings = predict_dict['embedding']
    label = gt_dict['inst_label']
    with tf.name_scope(scope, 'siamese_loss', [embeddings, label]):
        embedding_losses = []
        for i, (kernel_size, strides, padding, dilation_rate) in enumerate(
                zip(loss_config['kernel_size'], loss_config['strides'],
                    loss_config['padding'], loss_config['dilation_rate'])):
            embedding_pos_loss, embedding_neg_loss = dense_siamese_loss(
                embeddings,
                gt_dict['inst_label'],
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
            utils.summary_scalar('embedding_loss_{}'.format(i), embedding_loss)
        embedding_loss = tf.add_n(embedding_losses)
    return embedding_loss


def pixel_neighbor_binary_loss(loss_config, predict_dict, gt_dict, scope=None):
    """Dense pixel neighbor instance loss.
    Whether center pixel is from the same instance with its surroundings.

    Args:
        logits: A [b, h, w, c] predict logits tensor.
        inst_label: A [b, h, w, 1] instance label tensor.
        scope: name of scope.

    Returns:
        A scalar of pixel neighbor instance loss.
    """
    if not loss_config['use']:
        return 0.0

    kernel_size = loss_config['kernel_size']
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    w_size = kernel_size[0] * kernel_size[1]
    center_index = int((w_size - 1) / 2)

    logits = predict_dict['seg_cls']
    inst_label = gt_dict['inst_label']
    with tf.name_scope(scope, 'pixel_neighbor_binary_loss',
                       [logits, inst_label]):
        label = tf.squeeze(
            ops.im2col(
                inst_label,
                kernel_size,
                loss_config['strides'],
                loss_config['padding'],
                loss_config['dilation_rate'],
                merge_spatial=False),
            axis=1)  # [b, w_size, oh, ow]
        label = tf.transpose(label, [0, 2, 3, 1])
        label_center = label[..., center_index]
        label = tf.concat(
            [label[..., :center_index], label[..., center_index + 1:]], axis=-1)
        label = tf.equal(label_center, label)

        cls_loss = loss_config[
            'weight'] * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label, logits=logits)
    return cls_loss


class EmbeddingModel(object, metaclass=ABCMeta):
    """Abstract base class for embedding model. """

    def __init__(self,
                 fusion_layers,
                 embedding_depth,
                 seg_branch_config,
                 feature_scope=None):
        self._fusion_layers = fusion_layers
        self._embedding_depth = embedding_depth
        self._feature_scope = feature_scope
        self._seg_branch_config = seg_branch_config
        self._end_points = {}

    @abstractmethod
    def build(self, preprocessed_img, is_training=True, scope=None):
        """Build network and extract final embedding.

        Args:
            preprocessed_img: A [b, h, w, 3] float32 tensor.
            is_training: Option for batch normalization, must be correctly set.
            scope: name scope.

        Returns:
            A dict contains at least following field:
            1) embedding: A [b, ow, ow, c] tensor, final embedding of input
                images.
            may contains following field:
            1) seg_cls: logits for dense prediction.
        """
        pass

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Preprocess for each backbone network. """
        pass

    def get_layer(self, name):
        """Get specified feature given name. """
        return self._end_points['{}/{}'.format(self._feature_scope, name)]

    def set_layer(self, name, val):
        """Add layer to collection. """
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

    def loss(self, loss_config, predict_dict, gt_dict):
        """Compute loss for embedding model. """
        with tf.name_scope('loss'):
            embedding_loss = multi_scale_dense_siamese_loss(
                loss_config['dense_siamese_loss'], predict_dict, gt_dict)
            cls_loss = pixel_neighbor_binary_loss(
                loss_config['pixel_neighbor_binary_loss'], predict_dict,
                gt_dict)
            l2_loss = weight_l2_loss(loss_config['weight_decay'])
            total_loss = embedding_loss + l2_loss + cls_loss
            utils.summary_scalar('l2_loss', l2_loss)
            utils.summary_scalar('cls_loss', cls_loss)
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
