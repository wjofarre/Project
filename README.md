# Project
Project

```
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:26:46 2015

@author: liamofarrell
"""

# DAT7 project
import pandas as pd

# crime data
crime = pd.read_csv('chicago_crime.csv',na_filter=False)


 
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
for i in xrange(len(crime)):
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

obs = [int(x) for x in range(len(crime))]

crime['obs'] = obs

crime.distance.dropna()
crime.Budget.dropna()

crime['distance'] = crime.distance.astype(float)
crime['Budget'] = crime.Budget.astype(float)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['distance']
X = crime[feature_cols]
y = crime.obs
linreg.fit(X,y)
pred = linreg.predict(X)

plt.scatter(crime.distance, crime.obs)
plt.plot(crime.distance, pred, color='red')

print linreg.intercept_
print linreg.coef_
linreg.predict(50000)

y_pred = linreg.predict(X)
metrics.r2_score(y, y_pred)

zip(feature_cols, linreg.coef_)

y_true = [100, 50, 30, 20]
y_pred = [90, 50, 50, 30]

print metrics.mean_squared_error(y_true, y_pred)

print metrics.mean_squared_error(y_true, y_pred)
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))


crime.groupby('obs').Month.value_counts()

from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

treereg = DecisionTreeRegressor(max_depth=1, random_state=1)
scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

max_depth_range = range(1, 8)

RMSE_scores = []

for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')

```
