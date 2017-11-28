from pprint import pprint
import numpy as np
import tensorflow as tf
import ops
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

class OpsTestCase(tf.test.TestCase):

    def test_im2col_forward(self):
        
        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (1, 1)
        dilation_rate = (1, 1)

        x = np.arange(1, 17).reshape([1, 4, 4, 1]).astype(np.float32)
        x = np.concatenate([x, x], axis=0)
        x = np.concatenate([x, x], axis=-1)
        z = ops.im2dis(x, kernel_size, strides, padding, dilation_rate)
        with self.test_session() as sess:
            z_np = sess.run(z)
            self.assertAllClose(z_np[0,...],z_np[1,...])
            self.assertTupleEqual(z_np.shape, (2, 4, 4, 9))

        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (1, 1)
        dilation_rate = (2, 2)
        x = np.arange(1, 26).reshape([1, 5, 5, 1]).astype(np.float32)
        x = np.concatenate([x, x], axis=-1)
        z = ops.im2dis(x, kernel_size, strides, padding, dilation_rate)

        target = np.array([[[14,14,14,14],[16,16,16,4]],[[24,20,16,24],[24,20,16,4]]])
        

        with self.test_session() as sess:
            z_np = sess.run(z)
            self.assertAllClose(z_np[0, :2, :2, :4], target)
    
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
    '''
    def test_im2col_grad(self):
        def aux():
            ''
            kernel_size = (3, 3)
            strides = (1, 1)
            padding = (1, 1)
            dilation_rate = (1, 1)
            t = np.arange(1, 26).reshape([1, 5, 5, 1]).astype(np.float64);
            #t = np.concatenate([t,t],axis = -1)
            x = tf.Variable(t,dtype = tf.float64)
            y = ops.im2dis(x, kernel_size, strides, padding, dilation_rate)
            
            with self.test_session():
                gd = tf.test.compute_gradient(x,
                        ops.get_shape_list(x),
                        y,
                        ops.get_shape_list(y),
                        delta=1e-2)
            np.set_printoptions(threshold=100000) 
            for a in gd:
                print (a)
                print('------------------------------')
               
            with self.test_session():
                error = tf.test.compute_gradient_error(
                        x,
                        ops.get_shape_list(x),
                        y,
                        ops.get_shape_list(y),
                        delta=1e-2)
                self.assertLess(error, 1e-5)
            
        
        kernel_size_list = [(3, 3)] * 3
        strides_list = [(1, 1)] * 3
        padding_list = [(1, 1), (2, 2), (5, 5)]
        dilation_rate_list = [(1, 1), (2, 2), (5, 5)]

        x = tf.Variable(np.random.normal(size=[2, 6, 7, 3]), dtype=tf.float64)

        def aux():
            for (kernel_size, strides, padding, dilation_rate) in zip(
                    kernel_size_list, strides_list, padding_list,
                    dilation_rate_list):
                y = ops.im2dis(x, kernel_size, strides, padding, dilation_rate)
                with self.test_session():
                    error = tf.test.compute_gradient_error(
                        x,
                        ops.get_shape_list(x),
                        y,
                        ops.get_shape_list(y),
                        delta=1e-5)
                    self.assertLess(error, 1e-7)
        
        with tf.device('/cpu:0'):
            aux()
        #with tf.device('/gpu:0'):
         #   aux()
        '''

if __name__ == "__main__":
    tf.test.main()
