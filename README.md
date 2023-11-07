# Implementation of Simple Linear Regression Model forPredicting the Marks Scored

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
```
OUTPUT:

# df.head:
![dfhead](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/04f1a590-2d23-4ecd-8965-1452b404da8e)





# df.tail():
![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/8f1902a2-9314-49f6-8080-38c2b30391f6)




# Array value of X:

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/9617c242-3a97-4e01-8f7a-e3ed020b5f99)


# Array value of Y:
![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/40bbddba-8456-40fc-a7c9-55b41ee38c16)



# Values of Y prediction:

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/cf0d401f-8738-436b-b438-e4de83e91c76)



# Array values of Y test:

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/023c45ea-6fbe-455d-8e8d-5a7b8cf2ebef)




# Training Set Graph: 

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/bb60cc5e-5fe6-4abf-b247-51672abd4905)



# Test Set Graph:

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/513728a6-4072-4975-994a-6343eac54dc0)



# Values of MSE, MAE and RMSE:

![image](https://github.com/Sangavi-suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541861/ecaabc16-6e96-43ad-b7c8-ab2bfba1880b)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
