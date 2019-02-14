#%% [markdown]
# # Prepare data for LSTM 
#
# ##Lab 1
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
# ##Lab 2
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