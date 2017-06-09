# import tensorflow as tf 
# import numpy as np 
import matplotlib.pyplot as plt 
# import xlrd

# data_file = "/home/omar/Downloads/slr05.xls"
# # Step1: Start reading the input data 
# book = xlrd.open_workbook(data_file, encoding_override="utf-8")
# sheet = book.sheet_by_index(0)
# data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
# n_samples = sheet.nrows - 1

# # Step2: create placeholders for input and labeled data 
# X = tf.placeholder(tf.float32,name="X")
# Y = tf.placeholder(tf.float32,name="Y") 
# # Step3: represent weight and bias  
# w = tf.Variable(0.0)
# b = tf.Variable(0.0)
# # Step4: represent predicted data related to input data using linear model
# y_pred = X * w + b 

# # Step5: represent loss function 
# loss  = tf.reduce_sum(tf.square(y_pred - Y)) # sum of the squares
# #optimizer 
# optimizer = tf.train.GradientDescentOptimizer(0.001)
# train = optimizer.minimize(loss)
# # start train the data 
# with tf.Session() as sess:
# 	sess.run(tf.initialize_all_variables())
# 	for i in range(100):
# 		for x,y in data:
# 			sess.run(train,{X:x,Y:y})
# 	curr_w , curr_b,curr_loss = sess.run([w,b,loss])



# import numpy as np
# import tensorflow as tf

# # Model parameters
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# # training data
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
# # training loop
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#   sess.run(train, {x:x_train, y:y_train})

# # evaluate training accuracy
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import tensorflow as tf
import xlrd
DATA_FILE = "/home/omar/Downloads/slr05.xls"
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
# Step 2: create placeholders for input X (number of fire) and label Y (number of

# X = tf.placeholder(tf.float32, name="X")
# Y = tf.placeholder(tf.float32, name="Y")
# # Step 3: create weight and bias, initialized to 0
# w = tf.Variable(0.0, name="weights")
# b = tf.Variable(0.0, name="bias")
# # Step 4: construct model to predict Y (number of theft) from the number of fire
# Y_predicted = X * w + b
# # Step 5: use the square error as the loss function
# loss = tf.reduce_sum(tf.square(Y - Y_predicted))
# # Step 6: using gradient descent with learning rate of 0.01 to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# with tf.Session() as sess:
# # Step 7: initialize the necessary variables, in this case, w and b
# 	sess.run(tf.initialize_all_variables())
# 	# Step 8: train the model
# 	for i in range(100): # run 100 epochs
# 		for x, y in data:
# 		# Session runs train_op to minimize loss
# 			sess.run(optimizer, feed_dict={X:x, Y:y})
# 			# Step 9: output the values of w and b
# 	w_value, b_value = sess.run([w, b])





# Model parameters
W = tf.Variable([0.0], tf.float32)
b = tf.Variable([0.0], tf.float32)
# Model input and output
X = tf.placeholder(tf.float32)
linear_model = W * X + b
Y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - Y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
# training data
DATA_FILE = "/home/omar/Downloads/slr05.xls"
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# training loop
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) # reset values to wrong

for i in range(80):
	for x,y in data:
		sess.run(train, {X:x, Y:y})	
		curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {X:x, Y:y})  


# evaluate training accuracy
k = []
h = []
m = []
o = []
for i,j in data:
	k.append(i)
	h.append(j)
	m = k * curr_W + curr_b
	o = m

	
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
plt.scatter(k,h)
plt.plot(k,o,color='g')
plt.grid()
plt.show()
		