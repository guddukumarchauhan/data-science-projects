# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 04:31:42 2022

@author: guddu kumar
"""
#advertisement sale prediction from an existing custumer using logistic regression 
import pandas as pd 
data=pd.read_csv(r"C:\Users\guddu kumar\Downloads\Day_3_ Dataset_DigitalAd_dataset.csv")
print(data.isna().sum())
print(data)
print(data.head())
X=data.iloc[:,: -1].values
Y=data.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)


age=int(input("enter the new customer's age:"))
sal=int(input("enter the new customer salary:"))
newcustmer=[[age,sal]]
result=model.predict(sc.transform(newcustmer))
print(result)
if result==1:
    print("custumer will buy ")
else:
    print("custumer will not buy ")


from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,y_pred)
print(acc *100,'%')





