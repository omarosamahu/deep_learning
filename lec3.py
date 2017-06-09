import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
import xlrd

data_file = "/home/omar/Downloads/slr05.xls"
# Step1: Start reading the input data 
book = xlrd.open_workbook(data_file, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step2: Create placeholders for n fires denoted by X and Y for number of theft
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
# Step3: initialize weights and biases 
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")
# Step4: create a model to predicted Y
Y_pred = X * w + b
# Step5: Define ur loss fun 
loss = tf.reduce_sum(tf.square(Y - Y_pred)

# Step6 : using gradient descent with learning rate 0.01 to minimize lossx = tf.train.GradientDescentOptimizer(0.01)

with tf.Session() as sess:
	# Step7:initialize ness vars
	sess.run(tf.initialize_all_variables())
	# Step8: let's train the fuckin model 
	for i in range(100): # run 100 iteration
		for x,y in data:
			sess.run(optimizer,{X:x,Y:y})
	#Step 9 out w and b vals
	print sess.run([w,b,loss])
	

	# print sess.run(optimizer,feed_dict={X:x,Y:y})
	# print sess.run(loss)
	# print w_val,b_val
