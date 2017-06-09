import numpy as np
import tensorflow as tf 

# Declare list of features 
# Real valued features
features = [tf.contrib.layers.real_valued_column("x",dimension=1)]
# Esitimator or classifier
est = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Insert the data 
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# Start train the data 
est.fit(input_fn)
#
print est.evaluate(input_fn=input_fn)