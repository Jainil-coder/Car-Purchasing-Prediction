import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

data  = pd.read_csv("D:\\Machine Learning\\Ads.csv")

data.head(10)

real_x = data.iloc[:,[2,3]].values
real_y = data.iloc[:,4].values

X_train, X_test, y_train, y_test = train_test_split(real_x, real_y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Logistic Regression 

LR = LogisticRegression(random_state = 0)
LR.fit(X_train, y_train)

pred_y = LR.predict(X_test)

pred_y

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred_y))

# KNN Classifier

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train, y_train)

y_pred = KNN.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))