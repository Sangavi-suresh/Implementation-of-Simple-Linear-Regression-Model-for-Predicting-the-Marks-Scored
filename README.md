# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model. 

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.




## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sangavi suresh
RegisterNumber:  212222230130
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

# splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# displaying predicted values
y_pred

plt.scatter(x_train,y_train,color='brown')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='purple')
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE= ",mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)

OUTPUT:

df.head:
![dfhead](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/9353e376-31cf-4201-a060-aa0223e079d9)

df.tail():
![dftail](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/073f28c8-552c-4038-9c4d-158c4693ca16)

Array value of X:
![array x](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/184599c2-6fe6-4e74-b725-7dbd65e6894a)

Array value of Y:
![array y](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/3cbbbfa6-bbd4-429c-927f-5000c44e9a96)

Values of Y prediction:
![y pred](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/936437bf-c607-4fb0-8c8b-c7cab87b5cd7)

Array values of Y test:
![ytest](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/5aaf12ca-39e4-45cd-811e-766252821b6e)

Training Set Graph:
![train graph](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/001bc23f-3074-4333-bb62-eab4bb740153)

Test Set Graph:
![testgraph](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/55fa225d-0bcb-4950-8af3-92fb541a3726)

Values of MSE, MAE and RMSE:
![msemae](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/51bfd0dd-47b9-4eeb-8d94-17e8ca9b8f57)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
