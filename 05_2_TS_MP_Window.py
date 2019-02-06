#%% [markdown]
# # Multilayer Perceptron Using the Window method
# Multiple time steps can be used to make the prediction for the next step
# This is called as *_window_* method.
#
# We use t, t-1, t-2 to predict t+1

#%%
import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
import math

#%%
df = pd.read_csv('data/airline_passenger.csv',
 usecols=[1], engine='python', skipfooter=3)

#%%
plt.plot(df)
plt.show()

#%%
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)

#%%
ds = df.values
ds = ds.astype('float32')

#%%
train_size = int(len(ds) * 0.67)
test_size = len(ds) - train_size
train, test = ds[0:train_size,:], ds[train_size:len(ds),:]
print(len(train), len(test))

#%% [markdown]
# Need to create dataset that look back which the numbner of previous time steps to use as input variables to predict next time period
# X - # of passengers at a given time t
# y - # of passengers at the t+1
# **Here, to define the window size as by using look_back =3**

#%%
def create_dataset(dataset, look_back = 1):
    data_X, data_y = [], []

    for i in range(len(dataset) - look_back -1):
        a = dataset[i:(i+look_back), 0]
        data_X.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_y)

#%%
look_back = 3
train_X, train_y = create_dataset(train, look_back)
test_X, test_y = create_dataset(test, look_back)

print(train_X[:5])
print(train_y[:5])

#%% [markdown]
# * 1 input 
# * 1st hidden with 12 node
# * 2nd hidden with 8 node
# * 1 output

#%%
model = Sequential()
model.add(Dense(12, input_dim = look_back, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

#%%
h = model.fit(train_X, train_y, epochs = 400, validation_split=0.3, batch_size = 2, verbose =2 )

#%%
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

#%%
# Estimate model performance
trainScore = model.evaluate(train_X, train_y, verbose=0)
testScore = model.evaluate(test_X, test_y, verbose=0)

#%%
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0]))) 

print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

#%% [markdown]
# In the previous approach, the score was
# Train Score: 531.71 MSE (23.06 RMSE)
# Test Score: 2355.06 MSE (48.53 RMSE)
# 
# The error was reduced compared to previous.  Average error on the training dataset was 23 passengers (in K) and the average error on the unseen test set was 48.


#%%
# generate predictions for training
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

#%%
# shift train predictions for plotting
train_predict_plt = np.empty_like(ds)
train_predict_plt[:, :] = np.nan
train_predict_plt[look_back:len(train_predict)+look_back, :] = train_predict

#%%
# shift test predictions for plotting
test_predict_plt = np.empty_like(ds)
test_predict_plt[:, :] = np.nan
test_predict_plt[len(train_predict)+(look_back*2)+1:len(ds)-1, :] = test_predict

#%%
# plot baseline and predictions
plt.plot(ds)
plt.plot(train_predict_plt)
plt.plot(test_predict_plt)
plt.show()

#%% [markdown]
# Blue = whole dataset; Red:  Training, Green: Prediction