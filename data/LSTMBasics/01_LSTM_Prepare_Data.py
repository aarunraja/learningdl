#%% [markdown]
# # Prepare data for LSTM 
#
# ## Lab 1
# Normalize Series Data

#%%
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

#%%
# define contrived series
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
series = Series(data)
print(series)
#%%
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
#%%
# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
#%%
# normalize the dataset and print
normalized = scaler.transform(values)
print(normalized)
#%%
# inverse transform and print
inversed = scaler.inverse_transform(normalized)
print(inversed)

#%% [markdown]
# ## Lab 2
# Standardize Series Data

#%%
from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt
#%%
# define contrived series
data = [1.0, 5.5, 9.0, 2.6, 8.8, 3.0, 4.1, 7.9, 6.3]
series = Series(data)
print(series)
#%%
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
#%%
# train the normalization
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
#%%
# normalize the dataset and print
standardized = scaler.transform(values)
print(standardized)
#%%
# inverse transform and print
inversed = scaler.inverse_transform(standardized)
print(inversed)

#%% [markdown]
# ## Lab 3
# Prepadding Sequence Data

#%%
from keras.preprocessing.sequence import pad_sequences
#%%
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
#%%
# pad sequence
padded = pad_sequences(sequences)
print(padded)

#%% [markdown]
# ## Lab 4
# Postpadding Sequence Data

#%%
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pad sequence
padded = pad_sequences(sequences, padding='post')
print(padded)

#%% [markdown]
# ## Lab 5
# Pre and Post truncating Sequence Data

#%%
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pre truncate sequence
pretrun= pad_sequences(sequences, maxlen=2)
print(pretrun)
#%%
# post truncate sequence
posttrun= pad_sequences(sequences, maxlen=2, truncating='post')
print(posttrun)