# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 18:08:24 2022

@author: guddu kumar
"""
 
import pandas as pd 
data=pd.read_csv(r"C:\Users\guddu kumar\Downloads\heart.csv")
print(data.isna().sum())
print(data.head())
X_corr=data.corr()
print(X_corr)
X=data.iloc[:,[2,5,6,7.8,9,10,11,12]].values
Y=data.iloc[:,-1].values
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x=sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=12)


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(X_train,Y_train)
score=model.score(X_test,Y_test)
print(score)


y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))









 







 






