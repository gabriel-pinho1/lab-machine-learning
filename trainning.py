import numpy as np;
import matplotlib.pyplot as plt;
import pickle
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

digits=load_digits()
k=0
k_max = 10
score_best = 0
f1_best = 0
k_f1 = 0
k_best = 0
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) #Separate the data into training and test for model evaluation

for i in range(1,k_max):

    k=i
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores=metrics.accuracy_score(y_test,y_pred)
    f1_scores = metrics.f1_score(y_test, y_pred, average='macro')
    # print(f" K ={k} accuracy ={scores}")
    if score_best< scores:
        score_best = scores
        k_best = k

    if f1_best < f1_scores:
            f1_best = f1_scores
            k_f1 = k
print(f"best score {score_best} for {k_best} neighbours")  
print(f"best f1 score {f1_best} for {k_f1} neighbours")  




