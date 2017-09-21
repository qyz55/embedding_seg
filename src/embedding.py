"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import sys
import os
from datetime import datetime
from PIL import Image

import numpy as np
import scipy.misc
import tensorflow as tf

import model
import utils
import ops

slim = tf.contrib.slim


# TO DO: Move preprocessing into Tensorflow
def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)
    in_ = image[:, :, ::-1]
    in_ = np.subtract(in_,
                      np.array(
                          (104.00699, 116.66877, 122.67892), dtype=np.float32))
    # in_ = tf.subtract(tf.cast(in_, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    in_ = np.expand_dims(in_, axis=0)
    # in_ = tf.expand_dims(in_, 0)
    return in_


# TO DO: Move preprocessing into Tensorflow
def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label


def load_vgg_imagenet(ckpt_path, scope_name="vgg16"):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if "conv" in v:
            vars_corresp[v] = slim.get_model_variables(
                v.replace("vgg_16", scope_name))[0]
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
    return init_fn


def _train(dataset,
           initial_ckpt,
           learning_rate,
           logs_path,
           max_training_iters,
           save_step,
           display_step,
           global_step,
           iter_mean_grad=1,
           batch_size=1,
           momentum=0.9,
           resume_training=False,
           config=None,
           finetune=1,
           test_image_path=None,
           ckpt_name="osvos"):
    """Train OSVOS

    Args:
        dataset: Reference to a Dataset object instance
        initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
        supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
        learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
        logs_path: Path to store the checkpoints
        max_training_iters: Number of training iterations
        save_step: A checkpoint will be created every save_steps
        display_step: Information of the training will be displayed every display_steps
        global_step: Reference to a Variable that keeps track of the training steps
        iter_mean_grad: Number of gradient computations that are average before updating the weights
        batch_size: Size of the training batch
        momentum: Value of the momentum parameter for the Momentum optimizer
        resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
        config: Reference to a Configuration object used in the creation of a Session
        finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
        test_image_path: If image path provided, every save_step the result of the network with this image is stored
    Returns:
    """
    model_name = os.path.join(logs_path, ckpt_name + ".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

    # Create the network
    fusion_layers = ['conv1', 'conv2', 'conv3']  # FIXME(meijieru): as param
    final_embedding, end_points = model.build_model(input_image, fusion_layers)
    fusion_embeddings = [end_points[key] for key in fusion_layers]
    fusion_embeddings.append(final_embedding)

    # Initialize weights from pre-trained model
    if finetune == 0:
        init_weights = load_vgg_imagenet(initial_ckpt)

    # Define loss
    with tf.name_scope('losses'):
        embedding_losses = []
        # TODO(meijieru): im2col may be used once instead of multiple
        # times for input_label.
        for i, embedding in enumerate(fusion_embeddings):
            embedding_pos_loss, embedding_neg_loss = ops.dense_siamese_loss(
                embedding, input_label)
            embedding_loss = embedding_pos_loss + embedding_neg_loss
            embedding_losses.append(embedding_loss)
            utils.summary_scalar('loss/embedding_pos_loss_{}'.format(i),
                                 embedding_pos_loss)
            utils.summary_scalar('loss/embedding_neg_loss_{}'.format(i),
                                 embedding_neg_loss)
            utils.summary_scalar('loss/embedding_loss_{}'.format(i),
                                 embedding_loss)
        total_loss = tf.add_n(embedding_losses) + tf.add_n(
            tf.losses.get_regularization_losses())
        utils.summary_scalar('loss/total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        utils.summary_scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(
                        grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            layer_lr = parameter_lr()
            grad_accumulator_ops = []
            for var_ind, grad_acc in grad_accumulator.items():
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(
                    grad_acc.apply_grad(
                        var_grad * layer_lr[var_name], local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in grad_accumulator.items():
                mean_grads_and_vars.append((grad_acc.take_grad(iter_mean_grad),
                                            grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(
                mean_grads_and_vars, global_step=global_step)
    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Log evolution of test image
    if test_image_path is not None:
        #  TODO(meijieru)
        pass

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    with tf.Session(config=config) as sess:
        print('Init variable')
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(
            logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            # Load pre-trained model
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights(sess)
            else:
                print('Initializing from specified pre-trained model...')
                # init_weights(sess)
                var_list = []
                for var in tf.global_variables():
                    var_type = var.name.split('/')[-1]
                    if 'weights' in var_type or 'bias' in var_type:
                        var_list.append(var)
                saver_res = tf.train.Saver(var_list=var_list)
                saver_res.restore(sess, initial_ckpt)
            step = 1
        print('Weights initialized')

        print('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                batch_image, batch_label = dataset.next_batch(
                    batch_size, 'train')
                image = preprocess_img(batch_image[0])
                label = preprocess_labels(batch_label[0])
                run_res = sess.run(
                    [total_loss, merged_summary_op] + grad_accumulator_ops,
                    feed_dict={input_image: image,
                               input_label: label})
                batch_loss = run_res[0]
                summary = run_res[1]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                print(
                    "{} Iter {}: Training Loss = {:.4f}".format(
                        datetime.now(), step, batch_loss),
                    file=sys.stderr)

            # Save a checkpoint
            if step % save_step == 0:
                if test_image_path is not None:
                    # TODO(meijieru)
                    pass
                    #  curr_output = sess.run(
                #  img_summary,
                #  feed_dict={
                #  input_image: preprocess_img(test_image_path)
                #  })
                #  summary_writer.add_summary(curr_output, step)
                save_path = saver.save(
                    sess, model_name, global_step=global_step)
                print("Model saved in file: %s" % save_path)

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print("Model saved in file: %s" % save_path)

        print('Finished training.')


def train_parent(dataset,
                 initial_ckpt,
                 supervison,
                 learning_rate,
                 logs_path,
                 max_training_iters,
                 save_step,
                 display_step,
                 global_step,
                 iter_mean_grad=1,
                 batch_size=1,
                 momentum=0.9,
                 resume_training=False,
                 config=None,
                 test_image_path=None,
                 ckpt_name="osvos"):
    """Train OSVOS parent network
    Args:
    See _train()
    Returns:
    """
    finetune = 0
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path,
           max_training_iters, save_step, display_step, global_step,
           iter_mean_grad, batch_size, momentum, resume_training, config,
           finetune, test_image_path, ckpt_name)


def train_finetune(dataset,
                   initial_ckpt,
                   supervison,
                   learning_rate,
                   logs_path,
                   max_training_iters,
                   save_step,
                   display_step,
                   global_step,
                   iter_mean_grad=1,
                   batch_size=1,
                   momentum=0.9,
                   resume_training=False,
                   config=None,
                   test_image_path=None,
                   ckpt_name="osvos"):
    """Finetune OSVOS
    Args:
    See _train()
    Returns:
    """
    finetune = 1
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path,
           max_training_iters, save_step, display_step, global_step,
           iter_mean_grad, batch_size, momentum, resume_training, config,
           finetune, test_image_path, ckpt_name)


def test(dataset, checkpoint_file, result_path, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 1
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the cnn
    with slim.arg_scope(osvos_arg_scope()):
        net, end_points = osvos(input_image)
    probabilities = tf.nn.sigmoid(net)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([
        v for v in tf.global_variables()
        if '-up' not in v.name and '-cr' not in v.name
    ])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for frame in range(0, dataset.get_test_size()):
            img, curr_img = dataset.next_batch(batch_size, 'test')
            curr_frame = curr_img[0].split('/')[-1].split('.')[0] + '.png'
            image = preprocess_img(img[0])
            res = sess.run(probabilities, feed_dict={input_image: image})
            res_np = res.astype(np.float32)[0, :, :, 0] > 162.0 / 255.0
            scipy.misc.imsave(
                os.path.join(result_path, curr_frame),
                res_np.astype(np.float32))
            print('Saving ' + os.path.join(result_path, curr_frame))
