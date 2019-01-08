#%% [markdown]
# # Gradient & Backpropagation

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%%
df = pd.read_csv('data/banknotes.csv')
df.head()

#%%
# count how many banknotes in each class
df['class'].value_counts()

#%% [markdown]
# We can also calculate the fraction of the larger class by dividing the first row by the total number of rows:

#%%
df['class'].value_counts()[0]/len(df)

#%% [markdown]
# The larger class amounts to 55% of the total, so we if we build a model it needs to have an accuracy superior to 55% in order to be useful.

#%%
import seaborn as sns

#%%
sns.pairplot(df, hue="class")

#%% [markdown]
# We can see from the plot that the two sets of banknotes seem quite well separable. In other words the orange and the blue scatters are not completely overlapped. This induces us to think that we will manage to build a good classifier.

#%%
from sklearn.ensemble import RandomForestClassifier

#%% [markdown]
# and let's create an instance of the model with default parameters:

#%%
model = RandomForestClassifier()

#%% [markdown]
# Now let's separate the features from labels as usual:

#%%
X = df.drop('class', axis=1).values
y = df['class'].values

#%% [markdown]
# and we are ready to train the model.

#%%
from sklearn.model_selection import cross_val_score

#%% [markdown]
# And then we run it with the model, features and labels as arguments. This function will return 3 values for the test accuracy, one for each of the 3 folds.

#%%
cross_val_score(model, X, y)