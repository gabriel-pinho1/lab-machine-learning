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
k_max = 20
score_best = 0
f1_best = 0
k_f1 = 0
k_best = 0
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) #Separate the data into training and test for model evaluation
accuracies = []
f1_scores = []
k_values = range(1,k_max)
for k in k_values:

    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    acc_scores=metrics.accuracy_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    accuracies.append(acc_scores)
    f1_scores.append(f1)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, label='Accuracy', marker='o')
plt.plot(k_values, f1_scores, label='F1 Score', marker='s')

plt.title('k-NN Performance on Test Set')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Score')
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.savefig('knn_metrics.png')