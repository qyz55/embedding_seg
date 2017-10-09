import argparse
import json
import os
import random
import tempfile
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
        "--input-base-dir", type=str, default='', help="Root dir of dataset.")
    parser.add_argument(
        "--sequence-name", type=str, default='', help="Name of input sequence.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default='experiments/output',
        help="Where to save the output.")
    parser.add_argument(
        "--num-visual-images",
        type=int,
        default=2,
        help="How many images to save.")
    parser.add_argument(
        "--detailed-summary-every",
        type=int,
        default=10,
        help="Save detailed summaries every often.")
    return parser.parse_args()


args = get_arguments()
with open(args.json_config, 'rb') as f:
    config = json.load(f)

tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

snapshot_dir = tempfile.mkdtemp(
    prefix='video_seg_{}_'.format(args.sequence_name))
print('Tempdir is: {}'.format(snapshot_dir))

img_dir = os.path.join('JPEGImages', '480p', args.sequence_name)
ann_dir = os.path.join('Annotations', '480p', args.sequence_name)
img_list = [
    os.path.join(img_dir, img)
    for img in os.listdir(os.path.join(args.input_base_dir, img_dir))
]
img_list.sort()
ann_list = [
    os.path.join(ann_dir, ann)
    for ann in os.listdir(os.path.join(args.input_base_dir, ann_dir))
]
ann_list.sort()
content = [' '.join([img, ann, ann]) for (img, ann) in zip(img_list, ann_list)]

eval_config = config['eval_config']
model_config = config['model_config']
train_config = eval_config['eval_finetune_config']
use_finetune = eval_config['finetune']

if use_finetune:
    checkpoint_dir = os.path.join(snapshot_dir, 'checkpoints')
    print('Finetune on special sequence')
    utils.copy_to(args.json_config, snapshot_dir)

    augment_config = train_config['augment_config']

    # Finetune the network
    input_config = utils.construct_dataset(
        args.input_base_dir, content[:1],
        os.path.join(snapshot_dir, 'finetune_data_list.txt'))
    with tf.name_scope("create_inputs_finetune"):
        dataset = ImageSegmentReader(
            input_config, augment_config, is_training=True)

    global_step = slim.create_global_step()
    embedding.train_finetune(
        dataset,
        model_config,
        train_config,
        checkpoint_dir,
        args.num_visual_images,
        1000000,  # only keep the last checkpoint
        args.detailed_summary_every,
        global_step)
    print('Finetune finished.')

print('Evaling.')
with tf.Graph().as_default():
    if use_finetune:
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    else:
        ckpt_path = train_config['restore_from']

    input_config = utils.construct_dataset(args.input_base_dir, content[1:],
                                           os.path.join(snapshot_dir,
                                                        'eval_data_list.txt'))
    with tf.name_scope("create_inputs_eval"):
        dataset = ImageSegmentReader(input_config, None, is_training=False)

    embedding.test(dataset, model_config, ckpt_path, args.output_dir)
