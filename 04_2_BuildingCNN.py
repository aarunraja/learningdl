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


#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%% [markdown]
# # L1:  Tensor

#%%
# Scalar is just numbers
scalar = 5
scalar

#%%
# Vector is list of numbers 1D.  Tensor of order 1.
v = np.array([1, 5, 3, 2])
v.shape
#%%
# Matrices is tensors of order 2
m = np.array([[1, 5, 3, 2],
              [0, 9, 3, 1]])
m.shape

#%% [markdown]
# ## L2
# ## Tensor
# The actual tensor starts with order 3

#%%
t = np.array([[[0, 1, 0],
              [5, 0, 2]],
              [[1, 2, 4],
               [8, 3, 1]]])
t.shape

#%%
img = np.random.randint(255, size=(4,4,3), dtype='uint8')
img

#%%
plt.figure(figsize=(5,5))
plt.subplot(221)
plt.imshow(img)
plt.title('All Channels combined')

#%%
plt.subplot(222)
plt.imshow(img[:, :, 0], cmap='Reds')
plt.title('Red channel')

#%%
plt.subplot(223)
plt.imshow(img[:, :, 1], cmap='Greens')
plt.title('Green channel')

#%%
plt.subplot(224)
plt.imshow(img[:, :, 2], cmap='Blues')
plt.title('Blue channel')

plt.tight_layout()

#%% [markdown]
# # Lab L3 - Building Convolutional Layer
#%%
from keras.layers import Conv2D
from scipy import misc

#%% [markdown]
# * Refer 04_2_1.png (Convolutional kernel)
# * Refer 04_2_2.png (Feature map)

#%%
# loading an example image from scipy.misc
img = misc.ascent()

#%%
plt.figure(figsize=(5,5))
plt.imshow(img, cmap = 'gray')

#%%
img.shape

#%%
# Convolutional Layers wants order-4 tensor as input
# 1 axis of length 1 for color channel
# 1 axis length 1 for dataset index
img_tensor = img.reshape((1, 512, 512, 1))

#%% [markdown]
# Usually filter size is 3x3, 5x5 or 7x7
# For this evaluation, let us use 11x11


#%%
from keras.models import Sequential
from keras.layers import Dense

#%%
model = Sequential()
# 1 filter, all weights to one
model.add(Conv2D(1, (11, 11), kernel_initializer='ones',
    input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
model.summary()

#%%
img_pred_tensor = model.predict(img_tensor)

#%%
img_pred = img_pred_tensor[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')

#%% [markdown]
# A convolution with a kernel will produce a new image, whose
# pixels will be combination of the original pixels in a receptive field
# and the values of the weights in the kernel.
# **This image is produced by learning by the network**
#
# Two additional arguments to be considered:  **padding** and **stride**

#%%
img_pred_tensor.shape

#%%
img_tensor.shape

#%% [markdown]
# ### Padding
#
# The convolved image is slightly smaller than original due to default padding value as 'valid'.
# * valid - no padding
# * same - pad to keep the same image size (Needed when you think pixels at the border has meaning information)
# **Refer 04_2_3.png

#%%
model = Sequential()
# 1 filter, all weights to one
model.add(Conv2D(1, (11, 11), 
    padding='same',
    kernel_initializer='ones',
    input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
padsame_img_tensor = model.predict(img_tensor)
padsame_img_tensor.shape

#%%
plt.imshow(padsame_img_tensor[0, :, :, 0], cmap='gray')

#%% [markdown]
# ### Stride
#
# stride (1, 1) means with no padding, the output image will lose one pixel on each side
# 
# increasing stride means skipping few piexels
# 
# applicable only image with higher resolution

#%%
model = Sequential()
# 1 filter, all weights to one
model.add(Conv2D(1, (11, 11), 
    strides = (5, 5),
    padding='same',
    kernel_initializer='ones',
    input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
stride5x5_img_tensor = model.predict(img_tensor)
stride5x5_img_tensor.shape

#%%
plt.imshow(stride5x5_img_tensor[0, :, :, 0], cmap='gray')

#%%
# Asymmerric strides
model = Sequential()
# 1 filter, all weights to one
model.add(Conv2D(1, (11, 11), 
    strides = (11, 5),
    padding='same',
    kernel_initializer='ones',
    input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
stride11x5_img_tensor = model.predict(img_tensor)
stride11x5_img_tensor.shape

#%%
plt.imshow(stride11x5_img_tensor[0, :, :, 0], cmap='gray')

#%% [markdown]
# # L4 - Pooling

#%%
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D

#%%
# MaxPooling2D
model = Sequential()
# 1 filter, all weights to one
model.add(MaxPooling2D(pool_size=(5,5),
    input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
img_pred = model.predict(img_tensor)[0, :, :, 0]
img_pred.shape

#%%
plt.imshow(img_pred, cmap='gray')

#%%
# GlobalMaxPooling2D
model = Sequential()
# 1 filter, all weights to one
model.add(GlobalMaxPooling2D(input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
img_pred_tensor = model.predict(img_tensor)
img_pred_tensor.shape

