#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 00:23:56 2020

@author: swayamkaul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing data
dataset = pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X=X[:,1:]

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)"""

#fitting multilinearRegression model to trainingset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test set result

Y_pred=regressor.predict(X_test)

#Building the oprtional model using Backward elemination

import statsmodels.api as sm
X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X=X[:,2:]
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,2,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()