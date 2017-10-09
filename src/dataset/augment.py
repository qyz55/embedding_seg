import tensorflow as tf
from dataset import utils


def image_scaling(img, class_label, inst_label):
    """Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
        img: Training image to scale.
        label: Segmentation mask to scale.
    """
    scale = tf.random_uniform(
        [1], minval=0.8, maxval=1.2, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    return utils.resize_all(img, class_label, inst_label, new_shape)


def image_mirroring(img, class_label, inst_label):
    """Randomly mirrors the images.

    Args:
        img: Training image to mirror.
        label: Segmentation mask to mirror.
    """
    distort_left_right_random = tf.random_uniform(
        [1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    class_label = tf.reverse(class_label, mirror)
    inst_label = tf.reverse(inst_label, mirror)

    return img, class_label, inst_label


def random_crop_and_pad_image_and_labels(image,
                                         class_label,
                                         inst_label,
                                         crop_h,
                                         crop_w,
                                         ignore_label=255):
    """Randomly crop and pads the input images.

    Padded position will be ignored when calculate loss.

    Args:
        image: Training image to crop/ pad.
        label: Segmentation mask to crop/ pad.
        crop_h: Height of cropped segment.
        crop_w: Width of cropped segment.
        ignore_label: Label to ignore during the training.
    """
    class_label = tf.cast(class_label, dtype=tf.float32)
    class_label = class_label - ignore_label
    inst_label = tf.cast(inst_label, dtype=tf.float32)
    inst_label = inst_label - ignore_label
    image = tf.cast(image, dtype=tf.float32)

    combined = tf.concat(axis=2, values=[image, class_label, inst_label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0,
                                                tf.maximum(
                                                    crop_h, image_shape[0]),
                                                tf.maximum(
                                                    crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_class_label_dim = tf.shape(class_label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 5])
    img_crop = tf.cast(combined_crop[:, :, :last_image_dim], dtype=tf.uint8)
    class_label_crop = combined_crop[:, :, last_image_dim:
                                     last_image_dim + last_class_label_dim]
    class_label_crop = class_label_crop + ignore_label
    class_label_crop = tf.cast(class_label_crop, dtype=tf.uint8)
    inst_label_crop = combined_crop[:, :,
                                    last_image_dim + last_class_label_dim:]
    inst_label_crop = inst_label_crop + ignore_label
    inst_label_crop = tf.cast(inst_label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    class_label_crop.set_shape((crop_h, crop_w, 1))
    inst_label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, class_label_crop, inst_label_crop
