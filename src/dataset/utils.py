from PIL import Image
import numpy as np
import tensorflow as tf

ResizeMethod = tf.image.ResizeMethod

# colour map
pascal_voc_colour_map = [
    (0, 0, 0),
    # 0=background
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128)
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
]


def resize_one_image(img, new_size, method=ResizeMethod.BILINEAR):
    """Wrapper for tensorflow image_resize for one image. """
    img = tf.image.resize_images(
        tf.expand_dims(img, axis=0), new_size, method=method)
    return tf.squeeze(img, axis=0)


def _batch_map(fun, batch_images):
    """Map for batch images. """
    n, h, w, c = batch_images.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        outputs[i] = fun(batch_images[i])
    return outputs


def decode_labels(masks, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      masks: result of inference after taking argmax with shape [b, h, w, 1].
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """

    def _plot_image(mask):
        h, w, _ = mask.shape
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for j_, j in enumerate(mask[:, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = pascal_voc_colour_map[k]
                else:
                    #  FIXME(meijieru): use palette
                    k %= num_classes
                    pixels[k_, j_] = pascal_voc_colour_map[k]
        return np.array(img)

    return _batch_map(_plot_image, masks[:num_images])
