from sklearn.datasets import load_digits # Ir buscar fotos de números desenhados à mão
from sklearn.model_selection import train_test_split # Ferramenta que divide as fotos em fotos de treino e de teste, de forma aleatória
from sklearn.neighbors import KNeighborsClassifier # Seleciona o algoritmo a ser usado
from sklearn import metrics # Vai buscar as métricas que permitem avaliar a qualidade do modelo
import matplotlib.pyplot as plt
import numpy as np

#Many more things in the scikit documentation - https://scikit-learn.org/stable/user_guide.html


digits=load_digits()
#print(digits.DESCR)

#print(digits.data)

#print(digits.data.shape)

#print(digits.feature_names)

#print(digits.target)

#print(digits.target_names)

#example = digits.data[100,:]
#example = np.reshape(example, (8,8))

#plt.imshow(example, cmap='gray')
#plt.show()

#print(digits.target[100])


#print(X_train.shape)
#print(X_test.shape)

#print(y_train.shape)
#print(y_test.shape)





#print(y_pred)

#print(y_test[0:20])
#print(y_pred[0:20])


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) #Separate the data into training and test for model evaluation

scores_vector = np.empty(20,)
matthews_corrcoef_vector = np.empty(20,)
index = np.arange(1, 21)

for i in index:

    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    scores=metrics.accuracy_score(y_test,y_pred)
    matthews_corrcoef = metrics.matthews_corrcoef(y_test,y_pred)
    scores_vector[i-1] = scores
    matthews_corrcoef_vector[i-1] = matthews_corrcoef

#print(np.max(scores_vector))
#print(np.max(matthews_corrcoef_vector))

scores_matrix = np.vstack([index, scores_vector])
matthews_corrcoef_matrix = np.vstack([index, matthews_corrcoef_vector])

max_position_scores = np.argmax(scores_matrix[1])
max_position_matthews = np.argmax(matthews_corrcoef_matrix[1])

ideal_k_scores = scores_matrix[0, max_position_scores]
max_score = scores_matrix[1, max_position_scores]
ideal_k_matthews = matthews_corrcoef_matrix[0, max_position_matthews]
max_matthews = matthews_corrcoef_matrix[1, max_position_matthews]

print('Ideal K value for the Accuracy Score \n')
print(ideal_k_scores)
print()
print('Maximum value of the Accuracy Score \n')
print(max_score)
print()
print('Ideal K value for the Matthews Coeficient \n')
print(ideal_k_matthews)
print()
print('Maximum value of the Matthews Coeficient\n')
print(max_matthews)
print()


plt.plot(scores_vector, label='Accuracy Score')
plt.plot(matthews_corrcoef_vector ,label='MCC')
plt.title('Metrics for evaluating the trained model')
plt.legend()
plt.savefig('Perfomance_metrics.png')

