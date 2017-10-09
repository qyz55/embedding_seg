import os
from PIL import Image
import numpy as np
import tensorflow as tf
from dataset.reader import ImageReader
from dataset import augment
from dataset import utils


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image
        /path/to/class/mask /path/to/inst/mask'.

    Returns:
      Three lists with all file names for images, class_masks and inst_masks,
        respectively.
    """
    f = open(data_list, 'r')
    images = []
    class_masks = []
    inst_masks = []
    for line in f:
        try:
            image, class_mask, inst_mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = class_mask = inst_mask = line.strip("\n")
        images.append(data_dir + '/' + image)
        class_masks.append(data_dir + '/' + class_mask)
        inst_masks.append(data_dir + '/' + inst_mask)
    return images, class_masks, inst_masks


def decode_label(path_tf):

    def aux(path):
        img = Image.open(path)
        return np.expand_dims(np.array(img), -1).astype(np.uint8)

    res = tf.py_func(aux, [path_tf], tf.uint8)
    res.set_shape([None, None, 1])
    return res


def read_images_from_disk(input_queue):
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.

    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_contents = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(img_contents, channels=3)
    class_label = decode_label(input_queue[1])
    inst_label = decode_label(input_queue[2])
    return img, class_label, inst_label


class ImageSegmentReader(ImageReader):
    """Generic Segmentation labeled ImageReader.
    This reader reads images and corresponding segmentation masks from the disk
    , and enqueues them into a TensorFlow queue.
    """

    def __init__(self, input_config, augment_config, is_training=True):
        """Initialise an ImageReader.

        Args:
            input_config: A dictionary contains specified configs for input.
            augment_config: A dictionary contains specified configs for data
                augmentation.
            is_training: Whether to shuffle data.
        """
        self.data_dir = input_config['data_dir']
        self.data_list = input_config['data_list']
        self.input_size = input_config['input_size']
        self.ignore_label = input_config['ignore_label']
        self.augment_config = augment_config
        self.is_training = is_training

        (self.image_list, self.class_label_list,
         self.inst_label_list) = read_labeled_image_list(
             self.data_dir, self.data_list)
        self.queue = tf.train.slice_input_producer(
            [self.image_list, self.class_label_list, self.inst_label_list],
            shuffle=is_training)
        self.image, self.class_label, self.inst_label = read_images_from_disk(
            self.queue)

    def __len__(self):
        return len(self.image_list)

    def _not_augmented_data(self):
        """Without data augmentation. """
        if self.input_size:
            image, class_label, inst_label = utils.resize_all(
                self.image, self.class_label, self.inst_label, self.input_size)
        return image, class_label, inst_label

    def _augmented_data(self):
        """Apply data augmentation. """

        img, class_label, inst_label = [
            self.image, self.class_label, self.inst_label
        ]
        if self.input_size is not None:
            h, w = self.input_size

            # Randomly scale the images and labels.
            if self.augment_config['random_scale']:
                img, class_label, inst_label = augment.image_scaling(
                    img, class_label, inst_label)

            # Randomly mirror the images and labels.
            if self.augment_config['random_mirror']:
                img, class_label, inst_label = augment.image_mirroring(
                    img, class_label, inst_label)

            # Randomly crops the images and labels.
            if self.augment_config['random_crop']:
                (img, class_label,
                 inst_label) = augment.random_crop_and_pad_image_and_labels(
                     img, class_label, inst_label, h, w, self.ignore_label)

            img, class_label, inst_label = utils.resize_all(
                img, class_label, inst_label, self.input_size)

        return img, class_label, inst_label
