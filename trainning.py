import numpy as np;
import matplotlib.pyplot as plt;
import pickle;
import pandas as pd;
import pandas as pd;
from sklearn.datasets import load_digits;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn import metrics;

digits=load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) #Separate the data into training and test for model evaluation



for i in range(1,11):

    k=i
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores=metrics.accuracy_score(y_test,y_pred)
    print(f" K ={k} accuracy ={scores}")
  