import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf 

# Initiate some values
# x = tf.constant([2,4],name="a")
# y = tf.constant([[2,1],[3,4]],name="b")
a = tf.Variable(10,name="vector")
b = tf.Variable([4,5],name="vector")
c = tf.multiply(a,b)
val=a.assign(100)
# init = tf.variables_initializer([a,b])
with tf.Session() as sess:
	 sess.run(a.initializer)
	 sess.run(a.assign_add(10))
	 print a.eval()
# con = tf.constant([1.0,2.0],name="omar")
# with tf.Session() as sess:
# 	writer = tf.summary.FileWriter('./graphs', sess.graph)
# 	print sess.graph.as_graph_def()

# # Do some operations
# w = tf.add(x,y)
# z = tf.multiply(x,y)
# op1 = tf.pow(w,z)
# op2 = tf.multiply(x,w)



# Create a session
# with tf.Session() as sess:
# 	# Heeeeey i will use TensorBoard
# 	writer = tf.summary.FileWriter('./graphs', sess.graph) 
# 	res = sess.run([w,z])
# 	print res


# g = tf.get_default_graph()
# k = tf.Graph()

# with g.as_default():
# 	a = 4
# 	b = 4
# 	op1 = tf.add(a,b)

# sess = tf.Session(graph=g)
# print sess.run(op1)
# sess.close()

# with k.as_default():
# 	b = 9
# 	a = 9
# 	op1 = tf.add(a,b)

# sess = tf.Session(graph=k)
# print sess.run(op1)
# sess.close()
# with sess:
# 	print sess.run(op1)



# with g.as_default():
# 	a = 2
# 	b =3 
# 	c = tf.add(a,b)


# sess = tf.Session(graph=g) #sesssion is run within the graph
# with sess:
# 	print sess.run(c)




# a = tf.constant([9])
# b = tf.constant([19])

# c = tf.add(a,b)


# with tf.Session() as session:
# 	res = session.run(c)
# 	print res 
# with tf.device('/CPU:2'):
# 	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
#   	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
#   	c = tf.matmul(tf.reshape(a,[1,6]),tf.reshape(b,[6,1]))

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print sess.run(c)
