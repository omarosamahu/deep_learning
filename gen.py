#Generate Dataset
import matplotlib.pyplot as plt
import pylab
import numpy as np

x = np.linspace(-1,1,100)
sig = 2 + x + 2*x**2 
#Noisy data 
noise = np.random.normal(0,0.1,100)
y = sig + noise

plt.plot(sig,'b')
plt.plot(y,'g')
plt.plot(noise,'r')
plt.legend(["Without Noise","With Noise","Noise"],loc = 2)
plt.grid()
#Trained Data
x_train = x[0:80]
y_train = y[0:80]

#Model with degree 1 
plt.figure()
X_train = np.column_stack([np.power(x_train,i) for i in xrange(0,3)])
# mod = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train))))
# mod = np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),X_train.transpose,y_train)
t1 = np.linalg.inv(np.dot(X_train.transpose(),X_train))
t2 = X_train.transpose()

t3 = np.dot(t1,t2)
mod = np.dot(t3,y_train)
# plot the actual data
plt.plot(x,y,'g')
#Customize the predicted data
pred = np.dot(mod,[np.power(x,i) for i in xrange(0,3)])
#plot predicted data
plt.plot(x,pred,'r')
plt.grid()
pylab.legend(["Actual", "Predicted"], loc = 2)

#Let's calculate the mean square error
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - pred[0:80],
y_train - pred[0:80])))

test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - pred[80:],
y[80:] - pred[80:])))

print "Train RMSE (Degree = 2)" , train_rmse1
print "Test RMSE (Degree = 2)" , test_rmse1

#Model with degree 8 
plt.figure()
X_train = np.column_stack([np.power(x_train,i) for i in xrange(0,9)])
# mod = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train))))
# mod = np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),X_train.transpose,y_train)
t1 = np.linalg.inv(np.dot(X_train.transpose(),X_train))
t2 = X_train.transpose()

t3 = np.dot(t1,t2)
mod = np.dot(t3,y_train)
# plot the actual data
plt.plot(x,y,'g')
#Customize the predicted data
pred = np.dot(mod,[np.power(x,i) for i in xrange(0,9)])
#plot predicted data
plt.plot(x,pred,'r')
plt.grid()
pylab.legend(["Actual", "Predicted"], loc = 2)

#Let's calculate the mean square error
train_rmse2 = np.sqrt(np.sum(np.dot(y[0:80] - pred[0:80],
y_train - pred[0:80])))

test_rmse2 = np.sqrt(np.sum(np.dot(y[80:] - pred[80:],
y[80:] - pred[80:])))

print "Train RMSE (Degree = 8)" , train_rmse2
print "Test RMSE (Degree = 8)" , test_rmse2


plt.show()