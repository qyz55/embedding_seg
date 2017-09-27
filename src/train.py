"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import argparse
import json
import random
from pprint import pprint
import numpy as np
import tensorflow as tf
import utils
from core import embedding
from dataset.reader_seg import ImageSegmentReader

slim = tf.contrib.slim


def get_arguments():
    """Parse all the arguments provided from the CLI. """
    parser = argparse.ArgumentParser(description="FCN detection.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed to have reproducible results.")
    parser.add_argument(
        "--json-config",
        type=str,
        default='experiments/config/default.json',
        help="Where to load configs.")
    parser.add_argument(
        "--num-visual-images",
        type=int,
        default=2,
        help="How many images to save.")
    parser.add_argument(
        "--save-pred-every",
        type=int,
        default=2000,
        help="Save checkpoint every often.")
    parser.add_argument(
        "--detailed-summary-every",
        type=int,
        default=1000,
        help="Save detailed summaries every often.")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default='./experiments/checkpoints',
        help="Where to save snapshots of the model.")
    return parser.parse_args()


args = get_arguments()
utils.copy_to(args.json_config, args.snapshot_dir)
with open(args.json_config, 'rb') as f:
    config = json.load(f)
pprint(vars(args))
pprint(config)

tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

input_config = config['train_input_config']
train_config = config['train_config']
model_config = config['model_config']

augment_config = train_config['augment_config']

with tf.name_scope("create_inputs"):
    dataset = ImageSegmentReader(input_config, augment_config, is_training=True)

# Train the network
global_step = slim.create_global_step()
embedding.train_parent(
    dataset,
    model_config,
    train_config,
    args.snapshot_dir,
    args.num_visual_images,
    args.save_pred_every,
    args.detailed_summary_every,
    global_step)
