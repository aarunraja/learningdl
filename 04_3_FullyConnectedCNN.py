#%% [markdown]
# # Lab L5 - Fully Connected Network

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())
#%%
from keras.datasets import mnist

#%%
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%
X_train_flat = X_train.reshape((60000, 784))
X_test_flat = X_test.reshape(-1, 28*28)
X_train_sc = X_train_flat.astype('float32') / 255.0
X_test_sc = X_test_flat.astype('float32') / 255.0

#%%
from keras.utils.np_utils import to_categorical

#%%
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

#%%
# Reshaping as order-4 tensor
X_train_t = X_train_sc.reshape(-1, 28, 28, 1)
X_test_t = X_test_sc.reshape(-1, 28, 28, 1)

#%%
X_train_t.shape

#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation

#%% [markdown]
# * Convolution layer - 32 filter of size 3x3
# * Max pooling 2x2
# * ReLU activation
# * Couple of fully connected layer

#%%
model = Sequential()

model.add(Conv2D(32,
    (3, 3),
    input_shape=(28, 28, 1),
    kernel_initializer='normal'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

model.summary()

#%%
h = model.fit(X_train_t, y_train_cat, batch_size=128, 
    epochs=5, verbose=1, validation_split=0.3)

#%%
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

#%%
train_acc = model.evaluate(X_train_t, y_train_cat, verbose=0)[1]
test_acc = model.evaluate(X_test_t, y_test_cat, verbose=0)[1]

print("Training accuracy: {:0.4f}".format(train_acc))
print("Test accuracy: {:0.4f}".format(test_acc))