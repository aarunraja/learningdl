#%% [markdown]
# # CNN - Introduction
# Image is pixel that represented in grid of bindary value

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%% [markdown]
# binomial function to generate a 10x10 square matrix of random 0 and 1.

#%%
bw = np.random.binomial(1, 0.5, size=(10, 10))
bw

#%% [markdown]
# pyplot has imshow to vizualize it as an image

#%%
plt.imshow(bw, cmap='gray')
plt.title("Black and White pixels")

#%% [markdown]
# To create a grayscale image
# 8-bit resolution, 10x10 each integer value between 0 and 255
# Numpy's random.randint() generates random integers uniformly 
# distributed between a low and a high extremes.

#%%
gs = np.random.randint(0, 256, size=(10, 10))
gs
#%%
plt.imshow(gs, cmap='gray')
plt.title('Grey pixels')

#%% [markdown]
# ### MNIST dataset
# 70k images of 28x28 piexels each representing handwritten digit.
# target variable: 0 - 9
# As like sklit-learn, Keras has it's built-in MNIST dataset.

#%%
from keras.datasets import mnist

#%%
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%
print(X_train.shape)
print(X_test.shape)

#%% [markdown]
# The returned value is Numpy array with order 3.  3D matrix.
# select the first image in X_train

#%%
first_img = X_train[0]
first_img
#%%
plt.imshow(first_img, cmap='gray')

#%% [markdown]
# ## Pixels as features
# Numpy's reshape() - to reshape any array into a new shape

#%%
print(X_train.shape)
print(X_test.shape)

X_train_flat = X_train.reshape((60000, 784))
print(X_train_flat.shape)
X_test_flat = X_test.reshape(-1, 28*28)
print(X_test_flat.shape)

#%%
print(X_train_flat.min())
print(X_train_flat.max())

#%% [markdown]
# As we know, we can standardize 0..255 to 0..1
# Prior to this, we need to convert the integer to float32, then divide by 255.0

#%%
X_train_sc = X_train_flat.astype('float32') / 255.0
X_test_sc = X_test_flat.astype('float32') / 255.0

#%% [markdown]
# Now a fully connected Neural Network
# Goal is to multi-class for MNIST

#%%
y_train

#%% [markdown]
# Also check unique values in the y_train

#%%
np.unique(y_train)

#%% [markdown]
# In order to define a fully connected network, the shape of the label array should matches the output of the network
# the output is 10 output.
# We can use to_categorical()

#%%
from keras.utils.np_utils import to_categorical

#%%
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
print(y_train[0])
print(y_train_cat[0])
print(y_train_cat.shape)
print(y_test_cat.shape)

#%% [markdown]
# # Building the CNN
# Input: 784 features
# Layer 1: 512 nodes with Relu activation
# Layer 2: 256 nodes with Relu
# Layer 3: 128 nodes with Relu
# Layer 4: 32 nodes with Relu
# Output: 10 nodes with Softmax activation

#%%
from keras.models import Sequential
from keras.layers import Dense

#%%
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

model.summary()

#%% [markdown]
# Let's train for 10 epochs with 128 images per batch

#%%
h = model.fit(X_train_sc, y_train_cat, batch_size=128,
    epochs=10, verbose=1, validation_split=0.1)

#%%
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

#%% [markdown]
# The above chart shows validation does not perform well
# Let us evaluate the performance on test set

#%%
test_acc = model.evaluate(X_test_sc, y_test_cat)[1]
test_acc

#%%
train_acc = model.evaluate(X_train_sc, y_train_cat)[1]
train_acc

#%%
y_pred_cat = model.predict(X_test_sc)
y_pred = np.argmax(y_pred_cat, axis=-1)

#%%
print(y_pred[:5])
print(y_test[:5])

#%% [markdown]
# Train is better than test, means overfitting
# In this context, we need to provide better features from the images
# Any better way to extract features from images
# This process is called as feature extraction.
#
# ## Feature Extraction
# * Fourier transforms
# * Wavelet transforms
# * Histograms of oriented gradients (HOG)