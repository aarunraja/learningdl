#%% [markdown]
# # Gradient calculation in ANN

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%%
# x is 100 data points in [-4, 4]
x = np.linspace(-4, 4, 100)

# Define an abstract cost function
def J(w):
    return 70.0 - 15.0*w**2 + 0.5*w**3 + w**4

# Define derivative
def dJdw(w):
    return - 30.0*w + 1.5*w**2 + 4*w**3

#%% [markdown]
# Let's plot both functions:

#%%
plt.subplot(211)
plt.plot(x, J(x))
plt.title("J(w)")

plt.subplot(212)
plt.plot(x, dJdw(x))
plt.axhline(0, color='black')
plt.title("dJdw(w)")
plt.xlabel("w")

plt.tight_layout()

#%% [markdown]
# Now let's find the minimum value of $J(w)$ by gradient descent. The function we have chosen has two minima, one is a local minimum, the other is the global minimum. If we apply plain gradient descent we will stop at the minimum that is nearest to where we started. Let's keep this in mind for later.
# 
# Let's start from a random initial value of $w_0 = -4$:

#%%
w0 = -4

#%% [markdown]
# and let's apply the update rule:
# 
# $
# w_0 -> w_0 - \eta \frac{dJ}{dw}(w_0)
# $
# 
# We will choose a small learning rate of $\eta = 0.001$ initially:

#%%
lr = 0.001

#%% [markdown]
# The update step is:

#%%
step = lr * dJdw(w0)
step

#%% [markdown]
# and the new value of $w_0$ is:

#%%
w0 - step

#%% [markdown]
# i.e. we moved to the right, towards the minimum!
#%% [markdown]
# Let's do 30 iterations and se where we get:

#%%
iterations = 30

w = w0

ws = [w]

for i in range(iterations):
    step = lr * dJdw(w)
    w -= step
    ws.append(w)

ws = np.array(ws)

#%% [markdown]
# Let's visualize our descent, zooming in the interesting region of the curve:

#%%
plt.plot(x, J(x))
plt.plot(ws, J(ws), 'o')
plt.plot(w0, J(w0), 'or')
plt.legend(["J(w)", "steps", "starting point"])
plt.xlim(-4.2, -1);