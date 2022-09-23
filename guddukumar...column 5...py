# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 08:18:27 2022

@author: guddu kumar
"""

import pandas as pd 
data=pd.read_csv(r"C:\Users\guddu kumar\Downloads\50_Startups (2).csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
SI=SimpleImputer()

SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])


SI_Administration=SI.fit(data[['Administration']])
data['Administration']=SI_Administration.transform(data[['Administration']])


SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])


SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
data['State']=LB.fit_transform(data['State'])

print(data.corr()['Profit'])
data=data.drop('State',axis=1)

import matplotlib.pyplot as plt
plt.plot(data['R&D Spend'],data['Profit'])
plt.xlabel("R&D spend")
plt.ylabel("Profit spend")
plt.show()

plt.plot(data['Profit'],data['Marketing Spend'])
plt.xlabel('Profit spend')
plt.ylabel("Marketing Spend")
plt.show()

X=data.iloc[:,0:5].values
Y=data.iloc[:,-1].values
Y=Y.reshape(Y.shape[0],1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)



#print(linear regration ,polynomial regration,k nearest neighbor,support vector regration,decision tree regration)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)
Y1_pred=LR.predict(X_test)
from sklearn.metric import r2_score
score1=r2_score(Y_test ,Y1_pred)
print(score1*100,'%')

from sklearn.preprocessing import PolynomialFeatures
polynomial_ft = PolynomialFeatures(degree=2)
X_train=polynomial_ft.transform(X_train)

from sklearn.neighbors import KNeighborsRegressor
regration=KNeighborsRegressor(n_neighbors=5)
regration.fit(X_train ,Y_train)
y2_pred=regration.predict(X_test)
from sklearn.metrics import r2_score
score2=r2_score(Y_test,y2_pred)
print(score2*100,'%')

from sklearn.svm import SVR
svr_rbf=SVR(kernel='rbf')
svr_rbf.fit(X_train,Y_train)
y3_pred=svr_rbf.predict(X_test)
from sklearn.metrics import r2_score
score3=r2_score(Y_test,y3_pred)
print(score3*100,'%')

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion='mse')
regressor.fit(X_train,Y_train)
y4_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
score4=r2_score(Y_test,y4_pred)
print(score4*100,'%')




