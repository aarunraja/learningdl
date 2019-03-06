#%% [markdown]
# ## Decision Tree
# * Decision-tree is one of supervised learning algorithms.
# * It work for both classification and regression.
# * It can be used to visually and explicitly represent decision.
# * types of Decision tree
# ** Categorical Variable Decision Tree Eg: Student will play cricket or not
# ** Continuous Variable Decision Tree Eg: newal premium with an insurance company


# https://www.kaggle.com/drgilermo/playing-with-the-knobs-of-sklearn-decision-tree/notebook
# https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
# https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e236
#%%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#%%
#Read the data
train = pd.read_csv("./data/titanic/train.csv")
testset = pd.read_csv("./data/titanic/test.csv")

#%%
train.describe()
train.info()
#%%
testset.describe()
testset.info()

#%%
pd.isnull(train).sum() > 0

#%%
# Encode the sex feature to numeric
train['Sex'][train.Sex == 'female'] = 1
train['Sex'][train.Sex == 'male'] = 0
train.loc[:10,'Sex']

#%%
#Split the data to train and test sets:
columns = ['Pclass','Sex','Age','Fare','Parch','SibSp']
X = train[columns]
y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%%
clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)
print('Accuracy using the defualt gini impurity criterion...',clf.score(X_test,y_test))

clf = DecisionTreeClassifier(max_depth = 3, criterion = "entropy")
clf.fit(X_train,y_train)
print('Accuracy using the entropy criterion...',clf.score(X_test,y_test))

#%%
t = time.time()
clf = DecisionTreeClassifier(max_depth = 3, splitter = 'best')
clf.fit(X_train,y_train)
print('Best Split running time...',time.time() - t)
print('Best Split accuracy...',clf.score(X_test,y_test))

t = time.time()
clf = DecisionTreeClassifier(max_depth = 3, splitter = 'random')
clf.fit(X_train,y_train)
print('Random Split running time...',time.time() - t)
print('Random Split accuracy...',clf.score(X_test,y_test))

#%%
clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)
#http://www.webgraphviz.com/
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#%%
clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )