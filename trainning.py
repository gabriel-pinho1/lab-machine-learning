import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

digits=load_digits()
k=0
k_max = 10

k_statistics = {'k_number' : range(0,k_max),
                'best_result_count' : 0
                
                }
df = pd.DataFrame.from_dict(k_statistics)

for j  in range (25, 51): 
    size = j/100
    for n in range (1,50):
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=size, random_state=n) #Separate the data into training and test for model evaluation
        score_best = 0
        k_best = 1
        for i in range(1,k_max):

            k=i
            knn=KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train,y_train)
            y_pred=knn.predict(X_test)
            scores=metrics.accuracy_score(y_test,y_pred)
            # print(f" K ={k} accuracy ={scores}")
            if score_best< scores:
                score_best = scores
                k_best = k
        print(f"best score{score_best} for {k_best} neighbours for test_size: {size}")  
        # store result distribution
        df.loc[k_best, 'best_result_count'] += 1


