from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf


class ImageReader(object, metaclass=ABCMeta):
    """Abstract base class for image reader.  """

    def dequeue(self, num_elements, num_threads=4):
        """Pack images and masks into a batch.

        Args:
            num_elements: the batch size.
            num_threads: number of data producer. NOTE: For test, must be 1,
                else batch will be undeterministic.

        Returns:
            Images with shape [batch_size, h, w, 3].
            Class label with shape [batch_size, h, w, 1].
            Labels with shape [batch_size, h, w, 1].
        """
        with tf.name_scope('batch'):
            if self.is_training:
                image, class_label, inst_label = self._augmented_data()
            else:
                image, class_label, inst_label = self._not_augmented_data()

            image_batch, class_label_batch, inst_label_batch = tf.train.batch(
                [image, class_label, inst_label],
                num_elements,
                num_threads=num_threads)
            return image_batch, class_label_batch, inst_label_batch

    @abstractmethod
    def _not_augmented_data(self):
        """Without data augmentation. """
        pass

    @abstractmethod
    def _augmented_data(self):
        """Apply data augmentation. """
        pass

    @abstractmethod
    def __len__(self):
        pass
