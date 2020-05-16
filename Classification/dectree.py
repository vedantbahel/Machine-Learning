# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:05:20 2019

@author: bahel
"""

import numpy as np
import pandas as pd

df= pd.read_excel('Breat Cancer.xlsx')

X= df.iloc[:,2:].values
Y= df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test= train_test_split(X,Y, test_size=1/10, random_state=1)

from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(max_depth=10, random_state=101, max_features= None, min_samples_leaf=5)
dtree.fit(X, Y)
Y_pred= dtree.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y, Y_pred)


