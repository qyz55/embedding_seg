from PIL import Image
import tensorflow as tf
import numpy as np
from dataset.reader_seg import ImageSegmentReader
import dataset.utils as dutils
from core import ops


class ImageReaderSegTest(tf.test.TestCase):

    input_config = {
        "data_dir": "/data/VOCdevkit/VOC2012",
        "data_list": "../experiments/data/imageset/voc12/train/train.txt",
        "ignore_label": 255,
        "input_size": [161, 161]
    }

    def test_output_shape(self):
        batch_size = 4
        h, w = self.input_config['input_size']
        reader = ImageSegmentReader(self.input_config, None, is_training=False)
        image, cls_label, inst_label = reader.dequeue(batch_size, num_threads=1)
        with self.test_session() as sess:
            tf.train.start_queue_runners(sess=sess)
            image, cls_label, inst_label = sess.run(
                [image, cls_label, inst_label])
            self.assertTupleEqual(image.shape, (batch_size, h, w, 3))
            self.assertTupleEqual(cls_label.shape, (batch_size, h, w, 1))
            self.assertTupleEqual(inst_label.shape, (batch_size, h, w, 1))

    def test_visual(self):

        reader = ImageSegmentReader(self.input_config, {}, is_training=False)

        image, cls_label, inst_label = [
            reader.image, reader.class_label, reader.inst_label
        ]
        with self.test_session() as sess:
            tf.train.start_queue_runners(sess=sess)
            image, cls_label, inst_label = sess.run(
                [image, cls_label, inst_label])

            print(image.shape, cls_label.shape, inst_label.shape)
            img = Image.fromarray(image.astype(np.uint8))
            dst_path = 'origin.png'
            img.save(dst_path)
            print('Origin write to: {}'.format(dst_path))

            print(np.unique(cls_label))
            img = dutils.decode_labels(cls_label[None, ...])[0]
            img = Image.fromarray(img.astype(np.uint8))
            dst_path = 'cls_label.png'
            img.save(dst_path)
            print('Origin write to: {}'.format(dst_path))

            print(np.unique(inst_label))
            img = dutils.decode_labels(inst_label[None, ...])[0]
            img = Image.fromarray(img.astype(np.uint8))
            dst_path = 'inst_label.png'
            img.save(dst_path)
            print('Origin write to: {}'.format(dst_path))


if __name__ == "__main__":
    tf.set_random_seed(1234)
    tf.test.main()
