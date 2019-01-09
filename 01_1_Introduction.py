#%% [markdown]
# ## Tensorflow

#%%
import tensorflow as tf


#%%
# Declaring two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Creating symbolic expression using add function
add = tf.add(a, b)

#Creating a tensorflow session
session = tf.Session()


#%%
# Binding 1.5 to a and 2.5 to b
binding = {a: 1.5, b: 2.5}

# Executing
c = session.run(add, feed_dict=binding)
print(c)

#%% [markdown]
# ## PIMA Indians Diabetes - Keras

#%%
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#%% [markdown]
# Random seed helps to get same result when you run the model again

#%%
np.random.seed(6)

#%% [markdown]
# ### Step 1.  Load Data

#%%
df = np.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
X = df[:, 0:8]
y = df[:, 8]

#%% [markdown]
# ### Step 2. Define Model

#%%
model = Sequential()
# Input layer with 8 features and 12 neuron first hidden layer.
model.add(Dense(12, input_dim=8, activation='relu'))
# Second hidden layer with 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer.  Class 0 or 1.  Onset to diabetes within five years or not.
model.add(Dense(1, activation='sigmoid'))

#%% [markdown]
# ### Step 3. Compile Model
# Backend (Theano or TensorFlow) chooses the best way to represent the network for training and making predictions to run on your hardware.
# 
# Training - find best weights
# Compile requires additional parameters:
# 
# * Specifying loss function to evaluate a set of weights
# * Specifying optimizer used to search through different weights

#%%
# logarithmic loss for binary classification
# adam as gradient descent algorithm
# metrics helps to specify what is expected in this model, accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% [markdown]
# ### Step 4. Fit Model
# Execute the model on some data

#%%
from sklearn.model_selection import train_test_split


#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=6)


#%%
# training set X, y.  30% for validation as like as train_test_split
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)

#%% [markdown]
# ### Step 5. Evaluate the model

#%%
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#%%
y_test_pred = model.predict(X_test)
print(y_test[:3])
print(y_test_pred[:3])


