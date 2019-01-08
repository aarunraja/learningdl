#%% [markdown]
# ## Binary classification using DL

# Let us use make_moons which generate dataset for 
# binary classification in two interleaving moon 
# (swirl or two moons pattern)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 13)
pd.set_option('display.max_columns', 11)
pd.set_option("display.latex.repr", False)

from matplotlib.pyplot import rcParams

rcParams['font.size'] = 14
rcParams['lines.linewidth'] = 2
rcParams['figure.figsize'] = (8.4, 5.6)
rcParams['axes.titlepad'] = 14
rcParams['savefig.pad_inches'] = 0.15

#%%
from sklearn.datasets import make_moons

#%% [markdown]
# ## 1. Load Data
#%%
X, y = make_moons(n_samples=1000, noise=0.1, random_state = 6)

#%%
X.shape

#%%
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
df.head()
#%%
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.legend(['0', '1'])
plt.title('Non linearly separable data')

#%% [markdown]
# ## 2. Split the data set

#%%
from sklearn.model_selection import train_test_split

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

#%% [markdown]
# ## 3. Define model

#%% 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

#%%
model = Sequential()

#%% [markdown]
# Let us use Logistic Regression
# 2 inputs, 1 output node using sigmoid
# ![Model](assets/02_2_1.png)

#%%
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#%% [markdown]
# ## 4. Compile the Model
# Optimizer Adam - this is the algorithm that performs the actual learning.
# binary_crossentropy - cost function

#%%
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

#%% [markdown]
# ## 5. Fit the model

#%%
model.fit(X_train, y_train, epochs=200, verbose=0)

#%% 
results = model.evaluate(X_test, y_test)

#%% [markdown]
# ### Model Accuracy

#%%
print("The Accuracy score on the Test set is:\t",
      "{:0.3f}".format(results[1]))

#%% [markdown]
# Let us see the boundary identified by the logistic expression
#%%
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    c = model.predict(ab)
    cc = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
    plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    plt.legend(['0', '1'])
    
plot_decision_boundary(model, X, y)

plt.title("Decision Boundary for Logistic Regression")

#%% [markdown]
# As you can see in the figure, since a shallow model like logistic regression is not able to draw curved boundaries, the best it can do is align the boundary so that most of the blue dots fall in the blue region and most of the red crosses fall in the red region.

#%% [markdown]
# ## Deep model
# ![](assets/02_2_2.png)
# 3 layers.
# Input node - 2
# First hidden layer - 4 nodes (output ReLU)
# Second hidden layer - 2 nodes (output ReLU)
# Output - 1 node (sigmoid)

#%%
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)

#%% [markdown]
# predict_classes() to return actual predicted classes instead of predicted probability of each classes

#%%
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

#%%
y_test[:3]
#%%
y_test_pred[:3]
#%%
y_train_prob = model.predict(X_train)
y_test_prob = model.predict(X_test)

#%%
y_train_prob[:3]

#%% [markdown]
# ### Accuracy

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#%%
acc = accuracy_score(y_train, y_train_pred)
print("Accuracy (Train set):\t{:0.3f}".format(acc))

acc = accuracy_score(y_test, y_test_pred)
print("Accuracy (Test set):\t{:0.3f}".format(acc))

#%% [markdown]
# Let's plot the decision boundary for the model:

#%%
plot_decision_boundary(model, X, y)
plt.title("Decision Boundary for Fully Connected")

#%% [markdown]
# Optimize with different activation

#%%
model = Sequential()
model.add(Dense(4, input_dim=2, activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr=0.05),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, verbose=0)

plot_decision_boundary(model, X, y)


#%%
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)


#%%
acc = accuracy_score(y_train, y_train_pred)
print("Accuracy (Train set):\t{:0.3f}".format(acc))

acc = accuracy_score(y_test, y_test_pred)
print("Accuracy (Test set):\t{:0.3f}".format(acc))