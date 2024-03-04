# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]

    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta    

data=pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled=scaler.fit_transform(X1)
y1_scaled=scaler.fit_transform(y)
print(X)
print(X1_scaled)

theta=linear_regression(X1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction .reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")



Developed by: Yuvaraj B
RegisterNumber: 212222040186 
*/
```

## Output:
![linear regression using gradient descent](sam.png)
![Screenshot 2024-03-04 144352](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/997aacf1-ac01-43ac-ba0f-c9afeb63595d)
![Screenshot 2024-03-04 144405](https:/![Screenshot 2024-03-04 144423](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/98e632de-a768-4d68-8de5-012e322d116e)
![Screenshot 2024-03-04 144433](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/56a52eaa-b4c8-4d92-b028-2e57b485eee1)
![Screenshot 2024-03-04 144445](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/6ed6b8e7-d775-48ff-9e5b-d5fc8ce267b6)
![Screenshot 2024-03-04 144501](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/a5117f43-8665-4678-9196-c1ab39638905)
![Screenshot 2024-03-04 144514](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/e50e8d0a-93cc-4dcf-95f7-19f29b14cf98)


/github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/9d48bae4-0756-4824-a77b-5fa020749f44)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
