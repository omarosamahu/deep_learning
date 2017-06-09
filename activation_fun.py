import tensorflow as tf 
import numpy as np 

def sigmoid(z):
	return 1/np.exp(-z)
# Weights and baises construction 
W = tf.Variable([0.3],tf.float32)
b = tf.Variable([-0.3],tf.float32)
x = tf.placeholder(tf.float32)
# create a session 
sess = tf.Session()
# Initialize variables 
init = tf.initialize_all_variables()
sess.run(init)
# linear activation function
lin = W*x + b

# create another placeholder represents the output
y = tf.placeholder(tf.float32)
#square the error 
err = tf.square(lin - y)
#Define a loss function
loss = tf.reduce_sum(err)
# Training Data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# optimize the weights and biases 
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_model = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
	 sess.run(train_model,{x:x_train,y:y_train})
# print sess.run(err,{x:[1,2,3,4],y:[0,-1,-2,-3]})
# eval training accuracy 
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})

print "W:" , curr_W
print "b:" , curr_b
print "loss:" , curr_loss













# from theano import function
# import theano.tensor as T 
# import matplotlib.pyplot as plt
# import numpy as np 
# #sigmoid 
# a = T.dmatrix('a')
# f_a = T.nnet.sigmoid(a)
# f_sigmoid = function([a],[f_a])
# print "sigmoid: ",f_sigmoid([[-1,0,1]])
# # plt.plot(f_sigmoid([[-1,0,1]],np.linspace(-1,1,3)))
# # plt.grid()
# # plt.show()