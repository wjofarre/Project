# Project
Project

```
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:26:46 2015

@author: liamofarrell
"""

# DAT7 project
# Closest Distance

import pandas as pd

# crime data
crime = pd.read_csv('chicago_crime.csv')


 
# police station data   
stations = pd.read_csv('Police_Stations.csv', header=None)  


# haversine formula
from math import radians, sin, cos, sqrt, asin
 
def haversine(lat1, lon1, lat2, lon2):
 
  R = 6372.8 # Earth radius in kilometers
 
  dLat = radians(lat2 - lat1)
  dLon = radians(lon2 - lon1)
  lat1 = radians(lat1)
  lat2 = radians(lat2)
 
  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  c = 2*asin(sqrt(a))
 
  return R * c


with open('crime_coor.csv', 'rU') as f:
    crime_coor = [row.split(',') for row in f]



crime_lat = [float(row[0][0:]) for row in crime_coor]
crime_long = [float(row[1][0:]) for row in crime_coor]



with open('stations.csv', 'rU') as f:
    station_list = [row.split(',') for row in f]

station_lat = [float(row[0][0:]) for row in station_list]
station_long = [float(row[1][0:]) for row in station_list]



# calculate distnace from police station
nearest_pt = []
for i in xrange(len(crime_lat)):
    initial_dist = 1000
    for j in xrange(len(station_lat)):
        dist = haversine(crime_lat[i], crime_long[i],station_lat[j], station_long[j])
        if dist<initial_dist:
            initial_dist = dist
    nearest_pt.append(dist)
    
    
    
import matplotlib.pyplot as plt

# add distance to dataframe
crime['distance'] = nearest_pt

crime['Arrest'].value_counts()

crime.distance.plot(kind='box')

crime.boxplot(column='distance', by='Arrest')
  
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn.dummy import DummyRegressor

import seaborn as sns
import matplotlib.pyplot as plt


# haven't used this yet
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    

# map arrest true and false with numbers to graph them
crime['Arrest'] = crime.Arrest.map({1:1,0:0})

# logistic regression on arrest probabilty based on distance
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['distance']
X = crime[feature_cols]
y = crime.Arrest
logreg.fit(X, y)
pred = logreg.predict(X)

# graph data
plt.scatter(crime.distance, crime.Arrest)
plt.plot(crime.distance, pred, color='red')

# 
pred_prob = logreg.predict_proba(X)[:, 1]


plt.scatter(crime.distance, crime.Arrest)
plt.plot(crime.distance, pred_prob, color='red')

# testing predictabiilty
logreg.predict_proba(3)[:, 1]
```
