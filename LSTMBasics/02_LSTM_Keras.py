#%% [markdown]
# # Using Keras to define LSTM
#

#%%
import numpy as np

#%% [markdown]
# ## Lab 7
# Assume you have 1D or 2D dataset
# it should be transformed to tensor-3 using Numpy.reshape

#%%
data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data.shape

#%% [markdown]
# call reshape and passs it a tuple of the dimensions to which to transform your data.
# 
# In the 2-column data, let us treat the two columns as two time steps and reshape it as

#%%
# 1 sample, 10 time steps, 1 feature
data = data.reshape((1, 10, 1))
data.shape

#%% [markdown]
# If you like columns in your 2D data to become features with one time step, you can reshape it as

#%%
data2 = data.reshape((data.shape[0], 1, data.shape[1]))
data2.shape

#%% [markdown]
# You can specify the input_shape argument that expects a tuple containing the number of time steps and number of features.

#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

#%%
model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Dense(1))

#%% [markdown]
# Activation function that transform a summed signal from each neuron in a layer can be extracted and added to Sequential as a layer-like object called Activation.

#%%
model.add(Activation('sigmoid'))

#%% [markdown]
# Choice of activation function is important
#
# * Regression - linear (number of neurons matching number of outputs.  This is default)
# * Binary Classification - sigmoid
# * Multi-class classification - softmax.  One output neuron per class value,  assuming a one hot encoded output pattern.

#%%
model.compile(optimizer='sgd', loss='mse')

#%% [markdown]
# Optimizer can also be created like

#%%
from keras.optimizers import SGD
algo = SGD(lr=0.1, momentum=0.3)
model.compile(optimizer=algo, loss='mse')

#%%
model.fit(X, y, batch_size=32, epochs=100)
