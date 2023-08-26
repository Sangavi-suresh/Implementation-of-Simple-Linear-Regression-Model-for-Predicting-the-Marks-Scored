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


## Output:
df.head()

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/91c2016d-62c0-4e26-906c-e8917b266b3f)

df.tail()

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/d36ae374-f203-4b1e-a3b2-5aa5d3de130d)

Array value of X

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/624f3a14-847a-41c3-99d2-6e684b34b221)

Array value of Y

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/74cbdb19-0340-41d6-bab8-b25c0e4888f9)

Values of Y prediction

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/963d18f7-f2e1-4484-8981-05ccf5338557)

Array values of Y test

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/598d73c4-d7ad-427f-a070-60aba6d8d423)

Training Set Graph

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/e4d138b0-885b-47fa-a0d6-e7fd5361b3d7)

Test Set Graph


![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/0719643d-c920-4137-ad9d-7cc94d80186e)

Values of MSE, MAE and RMSE

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/ebf564de-af31-47a9-bab1-f25f7cecbbd8)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
