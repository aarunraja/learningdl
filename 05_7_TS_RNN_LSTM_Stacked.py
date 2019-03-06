#%% [markdown]
# # Stacked LSTM for Regression with Memory between batches
# LSTM networks can be stacked. return_sequences

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#%%
np.random.seed(7)
df = pd.read_csv('data/airline_passenger.csv',
 usecols=[1], engine='python', skipfooter=3)
ds = df.values
ds = ds.astype('float32')

#%% [markdown]
# ## Data Processing
# ### Rescaling
# LSTM are sensitive to the scale of the input data, specifically sigmoid or tanh.
# Hence, rescale to 0..1

#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#%%
scaler = MinMaxScaler(feature_range=(0, 1))
ds = scaler.fit_transform(ds)

#%% [markdown]
# ### Split the train and test
train_size = int(len(ds) * 0.67)
test_size = len(ds) - train_size
train, test = ds[0:train_size,:], ds[train_size:len(ds),:]
# reshape into X=t and y=t+1
look_back = 3

#%%
def create_dataset(dataset, look_back = 1):
    data_X, data_y = [], []

    for i in range(len(dataset) - look_back -1):
        a = dataset[i:(i+look_back), 0]
        data_X.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_y)

#%%
train_X, train_y = create_dataset(train, look_back)
test_X, test_y = create_dataset(test, look_back)

#%% [markdown]
# ## Reshaping
# LSTM expects input to be in 3-order tuple [samples, time steps, features]

#%% [markdown]
# Set the columns to be the time steps dimensions
# change the features dimension back to 1
#%%
# reshape input to [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

#%% [markdown]
# ## Building the LSTM model

#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#%% [markdown]
# ## Defining the model
# * 1 input
# * 2 hidden layer with 4 LSTM blocks
# * 1 output layer
#
# return_sequence=True means stacked

#%%
batch_size= 1
#%%
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True)) # note that return_sequences
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True)) # here not stacked
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#%%
for i in range(100):
    print("************* Outer Batch %d **********" % i)
    model.fit(train_X, train_y, epochs=10, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

#%%
# specify batch_size
train_predict = model.predict(train_X, batch_size=batch_size)
model.reset_states()
test_predict = model.predict(test_X, batch_size=batch_size)

#%%
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

#%%
train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))

#%%
print('Train Score: %.2f RMSE' % (train_score))
print('Test Score: %.2f RMSE' % (test_score))

#%%
# shift train predictions for plotting
train_predict_plot = np.empty_like(ds)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

#%%
# shift test predictions for plotting
test_predict_plot = np.empty_like(ds)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2)+1:len(ds)-1, :] = test_predict

#%%
# plot baseline and predictions
plt.plot(scaler.inverse_transform(ds))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.figure(figsize=(300, 200))
plt.show()

#%% [markdown]
# Compare the result.