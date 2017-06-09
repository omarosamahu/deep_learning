import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf 

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.matmul(a,b)
z = a * b 
dic = {a:[[1,2],[3,1]],b:3}
with tf.Session() as sess:
	print sess.run(z,dic) 