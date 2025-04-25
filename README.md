# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJITH.R
RegisterNumber: 212223230191

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### DATA HEAD
![424285256-f78e5f92-9dd3-4f39-8dae-a01edbfb1e08](https://github.com/user-attachments/assets/8a0c8da9-72e2-4c8c-a8fb-f0180cb4d7d0)
### DATA1 HEAD
![image](https://github.com/user-attachments/assets/6f4d01c4-d0a1-41db-b52b-6455655d4f88)
### ISNULL().SUM()
![image](https://github.com/user-attachments/assets/512427f5-1d30-4931-a24e-8cd8236ed4f5)
### DATA DUPLICATE
![image](https://github.com/user-attachments/assets/51b8b2fd-105c-4bb5-9e4f-53eb29121e11)
### PRINT DATA
![image](https://github.com/user-attachments/assets/72a69f14-f502-475d-aa06-4e0142b3700f)
### STATUS
![image](https://github.com/user-attachments/assets/dd5a910f-1dbc-456b-bffc-6170fff600f3)
### Y_PRED
![image](https://github.com/user-attachments/assets/97167f65-c6db-4cea-b493-e6d7bfdb7565)
### ACCURACY
![image](https://github.com/user-attachments/assets/f88f9672-71c6-4186-9d7b-f20b6fe77545)
### CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/fb7267e9-3500-4d1e-abff-478486852845)
### CLASSIFICATION
![image](https://github.com/user-attachments/assets/42b9cc70-b9c2-4dd1-8136-3a3bfb5fc5ef)
### LR PREDICT
![image](https://github.com/user-attachments/assets/c6b0f819-5fb9-4d50-b831-8e01312bf961)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
