# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 19:56:16 2018

@author: KC
"""

#Import Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#%matplotlib inline

#Import Dataset
dataset = pd.read_csv('HR.csv') 
X = dataset.iloc[:, [0,1,2,3,4,5,7,9]]
y = dataset.iloc[:, 6].values
dataset.head()

#Encoding Categorical Data(Salary)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 7] = labelencoder_X.fit_transform(X.iloc[:, 7])

#Spliting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Fitting Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0,oob_score=True)
classifier.fit(X_train, y_train)
print(classifier.oob_score_)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

#Scoring Metrics
import sklearn.metrics as metrics
ac = metrics.accuracy_score(y_test, y_pred)
print("\nAccuracy:",ac)
re = metrics.recall_score(y_test, y_pred, average='weighted')
print("\nRecall:",re)
pr = metrics.precision_score(y_test, y_pred, average='weighted')
print("\nPrecision:",pr)
f1=metrics.f1_score(y_test, y_pred, average='weighted')
print("\nF1 Score:",f1)

#Predictions
a = float(input("Satisfaction_level:"))
b = float(input("Evaluation:"))
c = int(input("No_of_projects:"))
d = int(input("Average_monthly_hours:"))
e = int(input("TimeSpentInCompany:"))
f = int(input("WorkAccident:"))
g = int(input("Promotion:"))
h = int(input("Salary(High->0,Low->1,Medium->2):"))
    
y1 = np.column_stack([[a],[b],[c],[d],[e],[f],[g],[h]])

ypredself = classifier.predict(y1)
print(ypredself)

#Correlation Matrix
corr = dataset.corr()
corr = (corr)
ax = plt.axes()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, ax=ax, cmap="YlGnBu")
ax.set_title('Heatmap of Correlation Matrix')
corr

#Turnover vs Salary
ax = plt.subplots(figsize=(8, 4))
sns.countplot(x="salary", hue='left', data=dataset).set_title('Employee Salary Turnover Distribution');

#Turnover vs Department
ax = plt.subplots(figsize=(12, 4))
sns.countplot(x="department", hue='left', data=dataset).set_title('Employee Department Turnover Distribution');

#Turnover vs Time_Spent_In_Company
ax = plt.subplots(figsize=(8, 4))
sns.countplot(x="time_spend_company", hue='left', data=dataset).set_title('Employee Time_Spent_In_Company Turnover Distribution');

#Turnover vs No_of_projects
ax = plt.subplots(figsize=(8, 4))
sns.countplot(x="number_project", hue='left', data=dataset).set_title('Employee No_of_projects Turnover Distribution');

#Employee Evaluation Distribution - Turnover V.S. No Turnover
fig = plt.figure(figsize=(12,5),)
ax=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'last_evaluation'] , color='b',shade=True,label='no turnover')
ax=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'last_evaluation'] , color='r',shade=True, label='turnover')
plt.title('Employee Evaluation Distribution - Turnover V.S. No Turnover')

#Employee Satisfaction Distribution - Turnover V.S. No Turnover
fig = plt.figure(figsize=(12,5),)
ax=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'satisfaction_level'] , color='b',shade=True,label='no turnover')
ax=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'satisfaction_level'] , color='r',shade=True, label='turnover')
plt.title('Employee Satisfaction Distribution - Turnover V.S. No Turnover')

#Employee Avg_monthly_hours Distribution - Turnover V.S. No Turnover
fig = plt.figure(figsize=(12,5),)
ax=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'average_montly_hours'] , color='b',shade=True,label='no turnover')
ax=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'average_montly_hours'] , color='r',shade=True, label='turnover')
plt.title('Employee Avg Monyhly hours Distribution - Turnover V.S. No Turnover')

#Avg monthly hrs Vs Satisfaction
sns.lmplot(x='satisfaction_level', y='average_montly_hours', data=dataset,fit_reg=False,hue='left')

#Evaluation Vs Avg monthly hours
sns.lmplot(x='average_montly_hours', y='last_evaluation', data=dataset,fit_reg=False,hue='left')

#Evaluation Vs Satisfaction
sns.lmplot(x='satisfaction_level', y='last_evaluation', data=dataset,fit_reg=False,hue='left')

# CONCLUSION

#Employees generally left when they are underworked (less than 150hr/month or 6hr/day)
#Employees generally left when they are overworked (more than 250hr/month or 10hr/day)
#Employees with either really high or low evaluations should be taken into consideration for high turnover rate
#Employees with low to medium salaries are the bulk of employee turnover
#Employees that had very less or very high project count was at risk of leaving the company
#Employee satisfaction is the highest indicator for employee turnover
#Employees with 4 and 5 years at a company are endangered of leaving
