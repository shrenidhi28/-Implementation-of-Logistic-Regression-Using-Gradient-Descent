# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Predict the values of array.

5. Calculate the accuracy. 


## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: C.Shrenidhi
RegisterNumber: 212223040196
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/content/Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)

dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes


dataset


X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,-1].values


theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
  return 1/(1+np.exp(-z))

def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h) +(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta -=alpha*gradient
  return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)


def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
print(y_pred)



accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew= predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)


*/
```

## Output:
## DATASET:
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/63147de0-f4af-4a54-8d09-958c653b3412)

## DATASET DTYPE:
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/eda15410-c3c0-4ba7-a512-dea59bbe0748)

 ## DATASET AFTER DROPPING:
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/713969df-868e-483d-b339-3c1e711a7847)

## Y PREDICT
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/4e6982fc-a988-48b9-95fd-cece4863de44)

## Y VALUE
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/a2561b99-7141-4800-b6fe-5f6c9a974fe5)

## ACCURACY
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/2864f8ae-1c0a-483e-a503-5b369d2bbaaa)

## Y PRED NEW
![image](https://github.com/shrenidhi28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155261096/6b9678fc-7e42-405b-a666-d321dda872cc)








## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

