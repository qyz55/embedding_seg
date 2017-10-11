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
import time
from datetime import datetime

import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image

import utils
import dataset.utils as dutils
from core import builder
from core import ops

slim = tf.contrib.slim


def _train(dataset,
           model_config,
           train_config,
           logs_path,
           num_visual_images,
           save_step,
           detailed_summary_step,
           global_step,
           resume_training=False,
           finetune=1):
    """Train model.

    Args:
        dataset: A dataset object instance.
        model_config: Config for model.
        train_config: Config for training.
        logs_path: Path to store the checkpoints.
        num_visual_images: Number of images to be visualized.
        save_step: A checkpoint will be created every save_steps.
        detailed_summary_step: Information of the training will be displayed every display_steps.
        global_step: Reference to a Variable that keeps track of the training steps.
        resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False).
        config: Reference to a Configuration object used in the creation of a Session.
        finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning.
    """
    # Prepare the input data
    image_batch, _, inst_label_batch = dataset.dequeue(
        train_config['batch_size'])

    # Create the network
    model = builder.build_model(model_config)
    final_embedding = model.build(
        model.preprocess(image_batch), is_training=True)

    # Define loss
    loss_config = train_config['loss_config']
    total_loss = model.loss(loss_config, final_embedding, inst_label_batch)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        total_loss = tf.identity(total_loss)

    # Define optimization method
    optimizer_config = train_config['optimizer_config']
    grad_clip_by_value = optimizer_config['grad_clip_by_value']
    all_variables = tf.trainable_variables()
    with tf.name_scope('optimize'):
        opt = builder.build_optimizer(optimizer_config, global_step)
        grads = tf.gradients(total_loss, all_variables)
        if grad_clip_by_value > 0.0:
            grads = [
                tf.clip_by_value(grad, -grad_clip_by_value, grad_clip_by_value)
                for grad in grads
            ]
        train_op = opt.apply_gradients(
            zip(grads, all_variables), global_step=global_step)

    for var in slim.get_model_variables():
        utils.summary_histogram('weight/{}'.format(var.op.name), var)
    for grad in grads:
        if grad is not None:
            utils.summary_histogram('grad/{}'.format(grad.op.name), grad)
    utils.summary_scalar('global_step', global_step)

    with tf.name_scope('visual'):
        visual_inst_label = tf.py_func(
            dutils.decode_labels, [inst_label_batch, num_visual_images, 21],
            tf.uint8,
            name='label')
        values = [
            image_batch[:num_visual_images], visual_inst_label,
            ops.embedding(final_embedding, num_save_images=num_visual_images)
        ]
        tf.summary.image('images', tf.concat(axis=2, values=values))

    # Log training info
    brief_summary = tf.summary.merge_all('brief')
    all_summary = tf.summary.merge_all()

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    model_name = os.path.join(logs_path, "model.ckpt")
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #  config.log_device_placement = True

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
        else:
            # Load pre-trained model
            initial_ckpt = train_config['restore_from']
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights = model.restore_fn(initial_ckpt)
            else:
                print('Initializing from specified pre-trained model...')
                init_weights = model.restore_fn(
                    initial_ckpt, from_embedding_checkpoint=True)
            init_weights(sess)

        print('Weights initialized')
        print('Start training')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        start_time = time.time()
        num_steps = optimizer_config['num_steps']
        while True:
            step = global_step.eval()
            if step > num_steps:
                print('Training process finished.')
                break

            if step % save_step == 0:
                save_path = saver.save(
                    sess, model_name, global_step=global_step)
                print("Model saved in file: %s" % save_path)

            if step % detailed_summary_step == 0:
                batch_loss, summary, _ = sess.run(
                    [total_loss, all_summary, train_op])
            else:
                batch_loss, summary, _ = sess.run([total_loss, brief_summary] +
                                                  [train_op])
            summary_writer.add_summary(summary, step)

            current_time = time.time()
            duration = current_time - start_time
            print(
                "{} Iter {}: Training Loss = {:.4f}, Duration: {}".format(
                    datetime.now(), step, batch_loss, duration),
                file=sys.stderr)
            start_time = time.time()

        save_path = saver.save(sess, model_name, global_step=global_step)
        print("Model saved in file: %s" % save_path)

        coord.request_stop()
        coord.join(threads)


def train_parent(dataset,
                 model_config,
                 train_config,
                 logs_path,
                 num_visual_images,
                 save_step,
                 detailed_summary_step,
                 global_step,
                 resume_training=False):
    """Train parent network

    Args:
        See _train()
    """
    finetune = 0
    _train(
        dataset,
        model_config,
        train_config,
        logs_path,
        num_visual_images,
        save_step,
        detailed_summary_step,
        global_step,
        resume_training=resume_training,
        finetune=finetune)


def train_finetune(dataset,
                   model_config,
                   train_config,
                   logs_path,
                   num_visual_images,
                   save_step,
                   detailed_summary_step,
                   global_step,
                   resume_training=False):
    """Finetune network.

    Args:
        See _train()
    """
    finetune = 1
    _train(
        dataset,
        model_config,
        train_config,
        logs_path,
        num_visual_images,
        save_step,
        detailed_summary_step,
        global_step,
        resume_training=resume_training,
        finetune=finetune)


def test(dataset,
         model_config,
         restore_from,
         result_path,
         loss_config=None):
    """Test one sequence.

    Args:
        dataset: Reference to a Dataset object instance.
        model_config: Config for model.
        restore_from: Path to checkpoint.
        result_path: Path to save the output images.
        loss_config: Config for loss.
    """
    # Prepare the input data
    image_batch, _, inst_label_batch = dataset.dequeue(1, num_threads=1)

    # Create the network
    model = builder.build_model(model_config)
    final_embedding = model.build(
        model.preprocess(image_batch), is_training=False)

    visual_embedding = ops.embedding(final_embedding, num_save_images=1)[0]
    fetches = [visual_embedding]

    if loss_config is not None:  # summary
        model.loss(loss_config, final_embedding, inst_label_batch)

        summary_writer = tf.summary.FileWriter(
            result_path, graph=tf.get_default_graph())
        all_summary = tf.summary.merge_all()
        fetches.append(all_summary)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #  config.log_device_placement = True
    with tf.Session(config=config) as sess:
        # Load pre-trained model.
        init_weights = model.restore_fn(
            restore_from, from_embedding_checkpoint=True)
        init_weights(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        num_tests = len(dataset)
        for i in range(num_tests):
            outputs = sess.run(fetches)
            if loss_config is not None:
                summary_writer.add_summary(outputs[1], i)

            visual_embedding_np = outputs[0]
            img = Image.fromarray(visual_embedding_np)
            img.save(os.path.join(result_path, 'embedding_{}.png'.format(i)))

        coord.request_stop()
        coord.join(threads)
