#%% [markdown]
# # Time Series Prediction using ANN

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%%
df = pd.read_csv('data/airline_passenger.csv',
 usecols=[1], engine='python', skipfooter=3)

#%%
df.describe()

#%%
df.head()

#%%
plt.plot(df)
plt.show()

#%% [markdown]
# ## Multilayer Perceptron Regression 
# Given the number of passengers this month, what is the number of passengers next month
#
# Need to convert our single column data into t and t+1

#%%
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)

#%%
ds = df.values
ds = ds.astype('float32')

#%% [markdown]
# usually we split the data into train and test.  In time series, sequence of value is important.  Hence, split the ordered dataset into train and test.
#
# Calculates the index of the split point and separate data into 67 - 33%

#%%
train_size = int(len(ds) * 0.67)
test_size = len(ds) - train_size
train, test = ds[0:train_size,:], ds[train_size:len(ds),:]
print(len(train), len(test))

#%% [markdown]
# Need to create dataset that look back which the numbner of previous time steps to use as input variables to predict next time period
# X - # of passengers at a given time t
# y - # of passengers at the t+1

#%%
def create_dataset(dataset, look_back = 1):
    data_X, data_y = [], []

    for i in range(len(dataset) - look_back -1):
        a = dataset[i:(i+look_back), 0]
        data_X.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_y)

#%%
look_back = 1
train_X, train_y = create_dataset(train, look_back)
test_X, test_y = create_dataset(test, look_back)

print(train_X[:5])
print(train_y[:5])

#%% [markdown]
# * 1 input
# * 1 hidden with 8 node
# * 1 output

#%%
model = Sequential()
model.add(Dense(8, input_dim = look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#%%
h = model.fit(train_X, train_y, epochs = 200, batch_size = 2, verbose =2 )

#%%
# Estimate model performance
trainScore = model.evaluate(train_X, train_y, verbose=0)
testScore = model.evaluate(test_X, test_y, verbose=0)

#%% [markdown]
# Taking the square root of the performance estimates, we can see that the model has an average error of 23 passengers (in thousands) on the training dataset and 48 passengers (in thousands) on the test dataset.

#%%
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore))) 

print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

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
# We can see that the model did a pretty poor job of fitting both the training and the test datasets. It basically predicted the same input value as the output. The plot makes the prediction look good, but in fact, the shift in the prediction results in a poor skill score.