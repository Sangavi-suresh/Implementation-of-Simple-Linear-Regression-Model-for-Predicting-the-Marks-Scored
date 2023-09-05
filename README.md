![ytest](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/59b0e671-aa3a-44c3-b9e8-0b48283ee568)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

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
![dfhead](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/6cdcc48b-3fab-495b-a1ab-e887ed08cd9a)


df.tail():
![dftail](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/57a122b5-50fe-4fb5-86b0-eecdd08a2cd6)


Array value of X:
![array x](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/3df672b2-1b65-485e-a394-d74e6df7b9d8)

Array value of Y:
![array y](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/195346fd-4a7f-4ab2-bc4b-8c3b5ce4a464)


Values of Y prediction:
![y pred](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/e2f65303-af4d-43a4-a7b7-95f01b4d038b)


Array values of Y test:
![ytest](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/54625bce-8ec2-4b1b-a954-58c57621d47d)



Training Set Graph:
![train graph](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/6925ca85-065a-4d11-9e80-b446a9ba0680)


Test Set Graph:
![testgraph](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/2781edfb-d4d6-4592-9280-6808c8784214)


Values of MSE, MAE and RMSE:
![msemae](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/2b862e16-12fb-4ad8-9b11-86cd851d143a)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
