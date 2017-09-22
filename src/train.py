"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import embedding
from reader import ImageReader

slim = tf.contrib.slim

# Training parameters
imagenet_ckpt = '../experiments/checkpoints/vgg_16.ckpt'
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
logs_path = '../experiments/checkpoints/embedding'
store_memory = True
max_training_iters_1 = 15000
max_training_iters_2 = 30000
max_training_iters_3 = 50000
save_step = 5000
test_image = None
display_step = 1
learning_rate = 1e-7
batch_size = 3

# Define Dataset
data_dir = '/data/VOCdevkit/VOC2012'
data_list = '../experiments/data/imageset/voc12/train/train.txt'
input_size = (320, 320)
random_scale = True
random_mirror = True
ignore_label = 255
IMG_MEAN = np.array(
    (104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

with tf.name_scope("create_inputs"):
    dataset = ImageReader(data_dir, data_list, input_size, random_scale,
                          random_mirror, ignore_label, IMG_MEAN)

# Train the network
global_step = slim.create_global_step()
embedding.train_parent(
    dataset,
    imagenet_ckpt,
    1,
    learning_rate,
    logs_path,
    max_training_iters_1,
    save_step,
    display_step,
    global_step,
    batch_size=batch_size,
    test_image_path=test_image,
    ckpt_name='embedding')
