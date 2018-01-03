import os
import ops
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
kernel_size = (3, 3)
strides = (1, 1)
padding = (1, 1)
dilation_rate = (1, 1)
t = np.arange(1, 26).reshape([1, 5, 5, 1]).astype(np.float64);
x = tf.Variable(t,dtype = tf.float64)
y = ops.im2dis(x, kernel_size, strides, padding, dilation_rate)
q = np.zeros(225).reshape([1, 5, 5, 9]).astype(np.float64);
q[0,0,0,7]=1
gra = tf.Variable(q,dtype = tf.float64)
gd = tf.gradients(y,x, grad_ys=gra)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
an = sess.run(gd)
print(an)
for i in an: 
    print(i[0,0,0,0])
