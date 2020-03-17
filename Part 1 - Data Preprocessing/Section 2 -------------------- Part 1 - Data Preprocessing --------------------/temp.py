#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:13:22 2020

@author: swayamkaul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values
#%%
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan,strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#%%
