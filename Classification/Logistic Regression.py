# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:46:04 2019

@author: bahel
"""

import numpy as np
import pandas as pd

df= pd.read_excel('Breat Cancer.xlsx')

X= df.iloc[:,2:34].values
Y= df.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test= train_test_split(X,Y, test_size=0, random_state=1)

from sklearn.linear_model import LogisticRegression

iterations = [100,70,200,250]

param_grid = dict(max_iter=iterations)

lr = LogisticRegression()

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(lr , param_grid, n_jobs=-1,verbose=5)
grid_result = grid.fit(X_train, Y_train)


Y_pred=grid.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)