import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\DELL\Downloads\Salary_Data.csv")

#feature selection(dependent as y and independent as x)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#spliting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

# compare predicted and actual salaries from the test set
comparsion = pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(comparsion)

#visualize the test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#finding m value
m_slope=regressor.coef_
print(m_slope)

#finding c value
c_intercept = regressor.intercept_
print(c_intercept)

#y=mx+c
#pred :1
y_12 = (m_slope*12) + c_intercept
print(y_12)

#pred :2
y_20 = (m_slope*20) + c_intercept
print(y_20)

#this  will give mean of entire dataframe
dataset.mean()

#this  will give mean of particular column
dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset['Salary'].mode()

dataset.var()

dataset['Salary'].var()

#standard deviation
dataset.std()

dataset['Salary'].std()


#cofficient of variation(cv)


from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

#corelation

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

#skewness

dataset.skew()
dataset['Salary'].skew()

#standard error
dataset.sem() 

dataset['Salary'].sem()

#z-score
import scipy.stats as stats

dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

#degree of freedom
#it gives  no.of individual variable
a= dataset.shape[0] #no.of rows
b= dataset.shape[1]#no.of columns

degree_of_freedom = a-b
print(degree_of_freedom)

y_mean = np.mean(y)# finding mean of dependent variable

#SSR
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
#SSE
SSE = np.sum((y-y_pred)**2)
print(SSE)

SST=SSR+SSE
print(SST)

#r**2

r_square= 1-(SSR/SST)
r_square

#training score
bias = regressor.score(X_train,y_train)
print(bias)

variance = regressor.score(X_test,y_test)
print(variance)

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train,regressor.predict(X_train))
test_mse = mean_squared_error(y_test,y_pred)

#pickling file
import pickle

filename = 'linear_regression_model.pkl'

#open a file in write-binary mode and dump the model
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
    
print("model has been pickled and save as linear_regression_model.pkl")


import os
os.getcwd()













