# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:07:26 2019

@author: bahel
"""

import numpy as np
import pandas as pd

df= pd.read_excel('Breat Cancer.xlsx')

X= df.iloc[:,2:].values
Y= df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test= train_test_split(X,Y, test_size=1/10, random_state=11)

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB(priors=None, var_smoothing=1e-04)
nb.fit(X_train, Y_train)
Y_pred= nb.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)