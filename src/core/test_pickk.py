import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ops

st = tf.Variable([[[[0],[0],[0],[0]],[[1],[1],[255],[2]],[[1],[1],[2],[2]],[[0],[1],[1],[2]]],[[[2],[2],[2],[2]] \
     ,[[4],[1],[1],[1]],[[1],[1],[0],[0]],[[0],[0],[0],[0]]]], dtype = tf.int32)
re, nb, si = ops.random_pick_k(st, 3, max_instance=4)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
an, nu, ss = sess.run([re, nb, si])
print (an)
print (nu)
print (ss)
