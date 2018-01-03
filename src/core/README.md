Test data:
kernel_size = (3,3)
strides = (1,1)
padding = (1,1)
dilation_rate = (1,1)
input = np.arange(1,26).reshape([1,5,5,1]).astype(np.float64)


using tf.gradients():
	run "python3 test.py"
	Assume gradients_y[0,0,0,7] = 1, others = 0. See what's in gradients_x[0,0,0,0]. 
	I think it should be -1, and the program did produce -1. 



using tf.test.compute_gradient():
	run "python3 ops_test.py"
	See Jacobian_theoretical[0, 7] and Jacobian_numerical[0, 7] which I think is corresponding to the test on tf.gradients() above.
	But the two Jacobian both produced 1.

What's more surprising is that sometimes I ran ops_test.py a lot of times and it produced different results on some positions every time.
I doubt I did wrong on memory allocation. 
