#%% [markdown]
# # Multiclass classification
#
# More than two labels.  For example, image is cat, dog, dove
# Selection of activation for multiclass classification requires attention due to mutually exclusive classes, independent classes
# E.g. organizing blog posts using tags (independent)
# Categorize movies into genre horror, action, thriller
#
# Use softmax function when dealing mutually exclusive classes

#%% [markdown]
# ## 0. Import Libraries
#%%
with open('common.py') as fin:
    exec(fin.read())
with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%% [markdown]
# ## 1. Load data
# Iris dataset

#%%
df = pd.read_csv('data/iris.csv')
df.head()

#%%
df.describe()

#%%
X = df.drop('species', axis=1)
X.head()

#%% [markdown]
# ## 3. Data Pre-processing
# Encoding class labels and create y
#%%
from sklearn.preprocessing import LabelEncoder

#%%
class_le = LabelEncoder()
y = class_le.fit_transform(df['species'].values)
y

#%% [markdown]
# To use the labels in ANN, we need to expand this into 3 binary dummy columns

#%%
from keras.utils import to_categorical


#%%
y_cat = to_categorical(y)

#%% [markdown]
# Let's check out what the data looks like by looking at the first 5 values:

#%%
y_cat[:5]

#%%
from sklearn.model_selection import train_test_split
#%%
X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size=0.22,
                     random_state=6, stratify=y)

#%% [markdown]
# ## 2. Define Model
# * 4 input features
# * No hidden layer
# * 3 output layer
# * softmax activation

#%% 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

#%%
model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
model.compile(Adam(lr=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%% [markdown]
# ## 3. Fit the model
#%%
model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=30, verbose=0)

#%% [markdown]
# ## 4. Evaluate the model
#%%
y_pred = model.predict(X_test)
y_pred[:5]

#%% [markdown]
# Which class does our network think each flower is? We can obtain the predicted class with the `np.argmax`, which finds the index of the maximum value in an array:

#%%
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

#%%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#%%
print(classification_report(y_test_class, y_pred_class))

#%%
cm = confusion_matrix(y_test_class, y_pred_class)

pd.DataFrame(cm, index = class_le.classes_,
             columns = ['pred_'+c for c in class_le.classes_])

#%%
plt.imshow(cm, cmap='Blues')

#%%
plt.scatter(X.loc[y==0,'sepal_length'],
            X.loc[y==0,'petal_length'])

plt.scatter(X.loc[y==1,'sepal_length'],
            X.loc[y==1,'petal_length'])

plt.scatter(X.loc[y==2,'sepal_length'],
            X.loc[y==2,'petal_length'])

plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.legend(class_le.classes_)
plt.title("The Iris Dataset")

#%%
import seaborn as sns
#%%
g = sns.pairplot(df, hue="species")
g.fig.suptitle("The Iris Dataset")

#%% [markdown]
# Homework - Deep Network
# Based on this let us perform optimization