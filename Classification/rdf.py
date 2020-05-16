# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:33:35 2019

@author: bahel
"""

import numpy as np
import pandas as pd

df= pd.read_excel('Breat Cancer.xlsx')

X= df.iloc[:,2:].values
Y= df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test= train_test_split(X,Y, test_size=1/2, random_state=1)

from sklearn.ensemble import RandomForestClassifier
rfm= RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=3, random_state=11, max_features=None, min_samples_leaf=20)
rfm.fit(X_train, Y_train)
Y_pred= rfm.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)


# Extract single tree
estimator = rfm.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',rounded = True, proportion = False, precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')