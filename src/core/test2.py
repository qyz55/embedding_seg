import tensorflow as tf
a = tf.placeholder(tf.float32)
b = a * 2
tf.summary.histogram('sbbb',b)
summary_writer = tf.summary.FileWriter('./su')
q = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range (10):
	a_val = [1,2,3,2]
	g = sess.run(q,feed_dict = {a:a_val})
	summary_writer.add_summary(g, global_step=i)

