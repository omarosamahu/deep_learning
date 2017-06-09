# import pylab
# import numpy
# import matplotlib.pyplot as plt 

# x = numpy.linspace(-1,1,100)
# signal = 2 + x + 2 * x * x
# noise = numpy.random.normal(0, 0.1, 100)
# y = signal + noise
# x_train = x[0:80]
# y_train = y[0:80]
# train_rmse = []
# test_rmse = []
# degree = 80
# lambda_reg_values = numpy.linspace(0.01,0.99,100)
# for lambda_reg in lambda_reg_values:
#     X_train = numpy.column_stack([numpy.power(x_train,i) for i in xrange(0,degree)])
#     model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(),X_train) +
# lambda_reg * numpy.identity(degree)),X_train.transpose()),y_train)
#     predicted = numpy.dot(model, [numpy.power(x,i) for i in xrange(0,degree)])
#     train_rmse.append(numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicted[0:80],
# y_train - predicted[0:80]))))
#     test_rmse.append(numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicted[80:],
# y[80:] - predicted[80:]))))
# plt.plot(lambda_reg_values, train_rmse)
# plt.plot(lambda_reg_values, test_rmse)
# plt.xlabel(r"$\lambda$")
# plt.ylabel("RMSE")
# plt.legend(["Train", "Test"], loc = 2)
# plt.show()
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-1,1,100)
signal = 2 + x + 2*x*x
noise = np.random.normal(0,0.1,100) 
y = signal + noise

x_train = x[0:80]
y_train = y[0:80]

train_rmse = []
test_rmse = []

deg = 80
lamd_reg_vals = np.linspace(0.01,0.99,100)

for lamd_reg in lamd_reg_vals:
	X_train = np.column_stack([np.power(x_train,i) for i in xrange(0,deg)])
	
	t0 = np.dot(X_train.transpose(),X_train) 
	t1 = lamd_reg*np.identity(deg)
	t2 = np.linalg.inv(t0 + t1)
	t3 = X_train.transpose()

	t4 = np.dot(t2,t3)
	mod = np.dot(t4,y_train)
	pred = np.dot(mod,[np.power(x,i) for i in xrange(0,deg)])
	train_rmse.append(np.sqrt(np.sum(np.dot(y[0:80]-pred[0:80],y_train[0:80]-pred[0:80]))))
	test_rmse.append(np.sqrt(np.sum(np.dot(y[80:]-pred[80:],y[80:]-pred[80:]))))

plt.plot(lamd_reg_vals,train_rmse)
plt.plot(lamd_reg_vals,test_rmse)
plt.xlabel(r"$\lambda$")
plt.ylabel("RMSE")
plt.legend(["Train","Test"],loc = 2)
plt.grid()
plt.show()

