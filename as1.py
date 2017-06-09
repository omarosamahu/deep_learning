import tensorflow as tf 

x = tf.constant([[0,-2,-1],[0,1,2]],tf.float32)
y = tf.zeros([2,3],tf.float32)

with tf.Session() as sess:
	# print sess.run(tf.cond(x > y , lambda:tf.add(x,y),lambda:tf.sub(x,y)))
	
	print sess.run(tf.equal(x,y))


