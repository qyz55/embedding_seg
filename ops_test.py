from pprint import pprint
import numpy as np
import tensorflow as tf
import ops


class OpsTestCase(tf.test.TestCase):

    def test_im2col_forward(self):
        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (1, 1)
        dilation_rate = (1, 1)

        x = np.arange(1, 17).reshape([1, 4, 4, 1]).astype(np.float32)
        x = np.concatenate([x, x], axis=0)
        x = np.concatenate([x, x], axis=-1)
        y = ops.im2col(x, kernel_size, strides, padding, dilation_rate)

        with self.test_session() as sess:
            y_np = sess.run(y)
            self.assertAllClose(y_np[..., 0], y_np[..., 1])
            self.assertAllClose(y_np[0, ...], y_np[0, ...])
            self.assertTupleEqual(y_np.shape, (2, 9, 16, 2))

        kernel_size = (2, 2)
        strides = (2, 2)
        padding = (0, 0)
        dilation_rate = (1, 1)

        x = np.arange(1, 17).reshape([1, 4, 4, 1]).astype(np.float32)
        y = ops.im2col(x, kernel_size, strides, padding, dilation_rate)
        target = np.array([[1, 3, 9, 11.], [2, 4, 10, 12.], [5, 7, 13, 15.],
                           [6, 8, 14, 16.]]).reshape([1, 4, 4, 1])

        with self.test_session() as sess:
            y_np = sess.run(y)
            self.assertAllClose(y_np, target)

        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (1, 1)
        dilation_rate = (2, 2)

        x = np.arange(1, 26).reshape([1, 5, 5, 1]).astype(np.float32)
        y = ops.im2col(x, kernel_size, strides, padding, dilation_rate)
        target = np.array([[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4],
                           [0, 6, 7, 0], [7, 8, 9, 12], [9, 10, 0, 14],
                           [0, 16, 17, 0], [17, 18, 19, 22], [19, 20, 0, 24]])

        with self.test_session() as sess:
            y_np = sess.run(y)
            self.assertAllClose(y_np[0, :, :4, 0], target)

    def test_device_forward(self):
        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (1, 1)
        dilation_rate = (1, 1)

        x = np.random.normal(size=[2, 6, 7, 3]).astype(np.float32)
        with tf.device('/cpu:0'):
            y_cpu = ops.im2col(x, kernel_size, strides, padding, dilation_rate)
        with tf.device('/gpu:0'):
            y_gpu = ops.im2col(x, kernel_size, strides, padding, dilation_rate)

        with self.test_session() as sess:
            lhs, rhs = sess.run([y_cpu, y_gpu])
            self.assertAllClose(lhs, rhs)

    def test_im2col_grad(self):
        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (1, 1)
        dilation_rate = (1, 1)

        x = tf.Variable(np.random.normal(size=[2, 6, 7, 3]), dtype=tf.float64)

        def aux():
            y = ops.im2col(x, kernel_size, strides, padding, dilation_rate)
            with self.test_session():
                error_cpu = tf.test.compute_gradient_error(
                    x,
                    ops.get_shape_list(x),
                    y,
                    ops.get_shape_list(y),
                    delta=1e-5)
                self.assertLess(error_cpu, 1e-7)

        with tf.device('/cpu:0'):
            aux()
        with tf.device('/gpu:0'):
            aux()


if __name__ == "__main__":
    tf.test.main()
