# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:35:06 2019

@author: Vedant Bahel

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data set
dataset= pd.read_csv('House Data.csv')
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,1].values

#Splitting the data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size= 1/3)

#Fitting Simple Linear Regression 
#This is called Model 
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)

##Predicting the test results
Y_pred= regressor.predict(X_test)

#Visualising the training set Results

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt/title('Price Vs Sqft Living(Training set)')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt/title('Price Vs. Sqft Living(Test set)')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()

#Giving External Data for Price Prediction
Y_manualtest= regressor.predict(4230)
print(Y_manualtest)

#if we want to take manual input from the user and then calculate the price
a=int(input('What is the Sqft living of your house?'))
useroutput= regressor.predict(a)
print(useroutput)