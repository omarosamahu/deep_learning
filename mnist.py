import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])

w = tf.Variable(tf.zeros([784,10]),name="weight")
b = tf.Variable(tf.zeros([10]),name="bias")
# Indicate predicted o/p
# Apply softmax
y_pred = tf.nn.tanh(tf.matmul(x,w)+ b)

# Training operation 
# True dist 
y_ = tf.placeholder(tf.float32,[None,10])
# Apply cross-entropy
# cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(y_pred),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
# sq = tf.reduce_mean(tf.square(y_ - y_pred))

# optimization process 
# Reduce error using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
  		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  	# Test the trained model
  	# Evaluate the accuracy of the model 
  	corr = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_,1))
  	acc  = tf.reduce_mean(tf.cast(corr,tf.float32))
  	print sess.run(acc,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
  	# print sess.run(corr,feed_dict={x:mnist.test.images,y_:mnist.test.labels})









# import time 
# import numpy as np 
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# # Step1: Read data 
# MNIST = input_data.read_data_sets("/data/mnist",one_hot=True)
# # Step2: Define some parameters 
# eta = 0.01
# batch_size =128
# n_epochs =25 
# # Step3: Create placeholders for trained and labeled data 
# # Weights are initialized to random vars with mean 0 and std 0.01
# X = tf.placeholder(tf.float32,[batch_size , 784],name="image")
# Y = tf.placeholder(tf.float32,[batch_size , 10],name="label")
# # Step4: Create weights and bias



# # Step5: predict Y from X , w and b 
# pred = tf.matmul(X,w) + b 
# # Step6: Define loss fun using softmax fun
# entropy = tf.nn.softmax_cross_entropy_with_logits(pred,Y)
# loss = tf.reduce_mean(entropy) #computes the mean over examples in the batch

# # Step7: let's train the fuckin data 
# optimizer = tf.train.GradientDescentOptimizer(eta)
# init = tf.initialize_all_variables()

# with tf.Session() as sess:
# 	sess.run(init)
# 	n_bat = int(MNIST.train.num_examples/batch_size) 
# 	for i in range(n_epochs):
# 		X_batch , Y_batch = MNIST.train.next_batch(batch_size)
# 		sess.run([optimizer,loss],feed_dict={X:X_batch, Y:Y_batch})
# 	curr_W, curr_b, curr_loss  = sess.run([w, b, loss], {X:X_batch, Y:Y_batch})  



