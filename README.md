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
crime = pd.read_csv('chicago_crime.csv',na_filter=False, low_memory=False)

 
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


crime.Budget.plot(kind='bar')


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


print logreg.intercept_
print logreg.coef_
 
pred_prob = logreg.predict_proba(X)[:, 1]


plt.scatter(crime.distance, crime.Arrest)
plt.plot(crime.distance, pred_prob, color='red')

obs = [int(x) for x in range(len(crime_lat))]

crime['obs'] = obs

crime.distance.dropna()
crime.Budget.dropna()

crime['distance'] = crime.distance.astype(float)
crime['Budget'] = crime.Budget.astype(float)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['Budget','distance']
X = crime[feature_cols]
y = crime.Arrest
linreg.fit(X,y)
pred = linreg.predict(X)

plt.scatter(crime.Budget, crime.obs)
plt.plot(crime.distance, pred, color='red')

plt.scatter(crime.Budget, crime.Arrest)
plt.plot(crime.Budget, pred, color='red')

print linreg.intercept_
print linreg.coef_


print metrics.mean_squared_error(y_true, y_pred)
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))


crime.Month.value_counts()




plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')


crime.Month.value_counts().plot(kind='bar')

crime.distance.describe()

crime.plot(kind='bar', x = 'Year', y = 'Budget')

from sklearn.dummy import DummyRegressor
dumb = DummyRegressor(strategy='mean')


dum = pd.get_dummies(crime['Type']=='ASSAULT').iloc[:,1:]
dum1 = pd.get_dummies(crime['Type']=='ARSON').iloc[:,1:]
july_dum = pd.get_dummies(crime['Month']=='July').iloc[:,1:]
june_dum = pd.get_dummies(crime['Month']=='June').iloc[:,1:]


crime['assault_dum'] = dum
crime['arson_dum'] = dum1
crime['june'] = june_dum
crime['july'] = july_dum

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['distance', 'Budget', 'assault_dum', 'arson_dum']
X = crime[feature_cols]
y = crime.Arrest
logreg.fit(X, y)
pred = logreg.predict(X)

print logreg.intercept_
print logreg.coef_

pred_prob = logreg.predict_proba(X)[:, 1]


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['Budget', 'assault_dum', 'arson_dum', 'june','july']
X = crime[feature_cols]
y = crime.obs
linreg.fit(X,y)
pred = linreg.predict(X)

print linreg.intercept_
print linreg.coef_




pd.scatter_matrix(crime)

plt.scatter(crime.distance, crime.obs)
plt.scatter(crime.distance, pred, color='red')
plt.ylim(0,400000)
plt.xlim(0,36)

crime.corr()

sns.heatmap(crime.corr())



sns.pairplot(crime, x_vars=['Budget','assault_dum','arson_dum', 'june', 'july'], y_vars='obs', size=6, aspect=0.7, kind='reg')


lm = smf.ols(formula='obs ~ Budget+july+june', data=crime).fit()


lm.params





crime.Month.value_counts().plot(kind='bar')


crime.Year.value_counts()


crime.groupby('Year').june.value_counts()
crime.groupby('Year').july.value_counts()



crime_year = [[2011,328474,1403611788,31848,31983],[2012,319367,1325902022,30991,31891],[2013,304363,1342953075,27183,28462],[2014,270302,1363742049,24957,26105], [2015,147324*2,1447421767,21872,22795]]

df = pd.DataFrame(crime_year)
df.columns = ['year','num','budget','june','july']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['budget', 'june', 'july']
X = df[feature_cols]
y = df.num
linreg.fit(X,y)
pred = linreg.predict(X)

print linreg.intercept_
print linreg.coef_

pd.scatter_matrix(df)


df.corr()

sns.heatmap(df.corr())

plt.scatter(df.year,df.num)
plt.scatter(df.year,pred,color='red')


sns.pairplot(df, x_vars=['budget','june','july'], y_vars='num', size=6, aspect=0.7, kind='reg')

lm = smf.ols(formula='num ~ budget+july+june', data=df).fit()


lm.params
lm.rsquared
y_pred = linreg.predict(X)
metrics.r2_score(y,y_pred)















































from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(treereg, X, y, cv=5, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

treereg = DecisionTreeRegressor(max_depth=1, random_state=1)
scores = cross_val_score(treereg, X, y, cv=5, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

max_depth_range = range(1, 10)

RMSE_scores = []

for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

treereg.fit(X,y)

plt.plot(max_depth_range, RMSE_scores)


from sklearn.cross_validation import cross_val_score
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

from sklearn.ensemble import BaggingRegressor
bagreg = BaggingRegressor(DecisionTreeRegressor(), n_estimators=500, bootstrap=True, oob_score=True, random_state=1)

# fit and predict
bagreg.fit(X, y)
y_pred = bagreg.predict(X)
y_pred




from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg


estimator_range = range(10, 310, 10)


RMSE_scores = []


for estimator in estimator_range:
    rfreg = RandomForestRegressor(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


```
