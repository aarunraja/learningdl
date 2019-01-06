#%% [markdown]
# # Activation functions

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
#%% [markdown]
# ## Sigmoid & Step

#%%
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def step(x):
    return x > 0

x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.plot(x, step(x))
plt.legend(['sigmoid', 'step'])
plt.title('Sigmoid & Step')

#%% [markdown]
# ## Tanh
# \begin{equation}
# y = tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}
# \end{equation}
# 
# The advantage of this is that negative values of the weighted 
# sum are not forgotten by setting them to zero, 
# but are given a negative weight. 
# In practice `tanh` makes the network learn much 
# faster than `sigmoid` or `step`.
# Numpy has tanh function.

#%%
x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.plot(x, step(x))
plt.plot(x, np.tanh(x))
plt.legend(['sigmoid', 'step', 'tanh'])
plt.title('Sigmoid, Step & Tanh')

#%% [markdown]
# ### ReLU (Rectifier Linear Unit)
#%% [markdown]
# \begin{equation}
# y = \begin{cases}
# x & {\text{if }} x > 0 \\
# 0 & {\text{otherwise}}
# \end{cases}
# \end{equation}
# 
# or simply:
# 
# \begin{equation}
# y = \max(0, x)
# \end{equation}

# 
# Originally motivated from biology, it has been shown to be very effective and it is probably the most popular activation function for deep Neural Networks. It offers two advantages.
# 
# 1. If it's implemented as an `if statement` (the former of the two formulations above), it's calculation is very fast, much faster than smooth functions like `sigmoid` and `tanh`. 
# 2. Not being bounded on the positive axis, it can distinguish between two large values of input sum, which helps back-propagation converge faster.

#%%
def relu_dbg(x):
    print(x)
    condition = x > 0
    print(condition)
    print(condition * x)
    return condition * x

x = np.linspace(-10, 10, 5)
relu_dbg(x)

#%%
def relu(x):
    condition = x > 0
    return condition * x

#%%
x = np.linspace(-10, 10, 100)
plt.plot(x, relu(x))
plt.title('relu');

#%% [markdown]
# ## SeLU (Scaled Exponential Linear Unit)
# recently developed

#%%
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    res = scale * np.where(x>0.0,
                           x,
                           alpha * (np.exp(x) - 1))
    return res


#%%
x = np.linspace(-10, 10, 100)
plt.plot(x, relu(x))
plt.plot(x, selu(x))
plt.legend(['relu', 'selu'])
plt.title('ReLU and SeLU');

#%% [markdown]
# When creating a deep network, we will use one of these activation functions _between_ one layer and the next, in order to make the Neural Network nonlinear. These functions are the secret power of Neural Networks: with nonlinearities at each layer they are able to approximate very complex functions.
#%% [markdown]
# ## Binary classification
#%% [markdown]
# Let's work through classifying a binary dataset using a Neural Network. We'll need a dataset to work with to train our Neural Network. Let's create an example dataset with two classes that are not separable with a straight boundary, and let's separate them with a fully connected Neural Network. First we import the `make_moons` function from Scikit Learn:
