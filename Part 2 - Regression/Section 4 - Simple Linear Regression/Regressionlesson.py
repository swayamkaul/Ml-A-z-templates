#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 01:11:20 2020

@author: swayamkaul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing data
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Precting Test set results
Y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience [Training set]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience [Training set]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()