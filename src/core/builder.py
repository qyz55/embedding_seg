import tensorflow as tf
import utils
from models.vgg import VggEmbeddingModel
from models.resnet import Resnet101EmbeddingModel
from models.resnet import Resnet50EmbeddingModel


def build_learning_rate(global_step, lr_config, scope=None):
    """Learning rate adjust. """
    base_lr = lr_config['base_lr']
    decay_steps = lr_config['decay_steps']
    decay_rate = lr_config['decay_rate']

    with tf.name_scope(scope, 'learning_rate'):
        prev = -1
        scale_rate = 1.0

        cases = []
        for decay_step in decay_steps:
            cases.append((tf.logical_and(global_step > prev,
                                         global_step <= decay_step),
                          lambda v=scale_rate: v))
            scale_rate *= decay_rate
            prev = decay_step
        cases.append((global_step > prev, lambda v=scale_rate: v))
        learning_rate_scale = tf.case(cases, lambda: 0.0, exclusive=True)
        res = learning_rate_scale * base_lr

    utils.summary_scalar('lr', res)
    return res


def build_optimizer(optimizer_config, global_step):
    opt_type = optimizer_config['type']
    lr_config = optimizer_config['learning_rate']
    if opt_type == 'SGD':
        learning_rate = build_learning_rate(global_step, lr_config)
        opt = tf.train.MomentumOptimizer(
            learning_rate,
            optimizer_config['momentum'],
            use_nesterov=optimizer_config['use_nesterov'])
    elif opt_type == 'Adam':
        base_lr = lr_config['base_lr']
        opt = tf.train.AdamOptimizer(base_lr, beta1=0.9, beta2=0.999)
    return opt


model_name_maps = {
    'vgg': VggEmbeddingModel,
    'resnet101': Resnet101EmbeddingModel,
    'resnet50': Resnet50EmbeddingModel
}


def build_model(model_config):
    cl = model_name_maps[model_config['model_name']]
    return cl(model_config['fusion_layers'], model_config['embedding_depth'])
