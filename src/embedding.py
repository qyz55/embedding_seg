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


def load_vgg_imagenet(ckpt_path, scope_name=None):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM

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


def _train(dataset,
           initial_ckpt,
           learning_rate,
           logs_path,
           max_training_iters,
           save_step,
           display_step,
           global_step,
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
        learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
        logs_path: Path to store the checkpoints
        max_training_iters: Number of training iterations
        save_step: A checkpoint will be created every save_steps
        display_step: Information of the training will be displayed every display_steps
        global_step: Reference to a Variable that keeps track of the training steps
        batch_size: Size of the training batch
        momentum: Value of the momentum parameter for the Momentum optimizer
        resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
        config: Reference to a Configuration object used in the creation of a Session
        finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
        test_image_path: If image path provided, every save_step the result of the network with this image is stored
    """
    model_name = os.path.join(logs_path, ckpt_name + ".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    image_batch, _, inst_label_batch = dataset.dequeue(batch_size)
    input_image = image_batch
    input_label = inst_label_batch

    # Create the network
    # FIXME(meijieru): as param
    kernel_size = (3, 3)
    strides = (1, 1)
    padding = (1, 1)
    dilation_rates = [(1, 1), (2, 2), (5, 5)]
    fusion_layers = [
        'vgg_16/conv1/conv1_2', 'vgg_16/conv2/conv2_2', 'vgg_16/conv3/conv3_3'
    ]
    final_embedding, end_points = model.build_model(input_image, fusion_layers)

    # Initialize weights from pre-trained model
    if finetune == 0:
        init_weights = load_vgg_imagenet(initial_ckpt)

    # Define loss
    with tf.name_scope('losses'):
        embedding_losses = []
        # TODO(meijieru): im2col may be used once instead of multiple
        # times for input_label.
        for i, dilation_rate in enumerate(dilation_rates):
            embedding_pos_loss, embedding_neg_loss = ops.dense_siamese_loss(
                final_embedding, input_label, kernel_size, strides, padding,
                dilation_rate)
            embedding_loss = embedding_pos_loss + embedding_neg_loss
            embedding_losses.append(embedding_loss)
            utils.summary_scalar('loss/embedding_pos_loss_{}'.format(i),
                                 embedding_pos_loss)
            utils.summary_scalar('loss/embedding_neg_loss_{}'.format(i),
                                 embedding_neg_loss)
            utils.summary_scalar('loss/embedding_loss_{}'.format(i),
                                 embedding_loss)
        embedding_loss = tf.add_n(embedding_losses)
        l2_loss = tf.add_n(tf.losses.get_regularization_losses())
        total_loss = embedding_loss + l2_loss
        utils.summary_scalar('loss/l2_loss', l2_loss)
        utils.summary_scalar('loss/embedding_loss', embedding_loss)
        utils.summary_scalar('loss/total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        utils.summary_scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(
            total_loss, var_list=tf.trainable_variables())
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        while step < max_training_iters + 1:
            print(step)
            batch_loss, summary, _ = sess.run([total_loss, merged_summary_op] +
                                              [train_op])
            #  batch_loss, summary = sess.run([total_loss, merged_summary_op])
            print('run')

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

        coord.request_stop()
        coord.join(threads)


def train_parent(dataset,
                 initial_ckpt,
                 supervison,
                 learning_rate,
                 logs_path,
                 max_training_iters,
                 save_step,
                 display_step,
                 global_step,
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
    _train(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters,
           save_step, display_step, global_step, batch_size, momentum,
           resume_training, config, finetune, test_image_path, ckpt_name)


def train_finetune(dataset,
                   initial_ckpt,
                   supervison,
                   learning_rate,
                   logs_path,
                   max_training_iters,
                   save_step,
                   display_step,
                   global_step,
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
    _train(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters,
           save_step, display_step, global_step, batch_size, momentum,
           resume_training, config, finetune, test_image_path, ckpt_name)


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
