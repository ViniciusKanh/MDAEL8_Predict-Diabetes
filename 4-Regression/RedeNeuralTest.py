import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import imblearn
from imblearn.over_sampling import SMOTE


# Load the dataset
input_file = '0-Datasets/dados_diabetes_normalizados menos valores faltantes.data'
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
df = pd.read_csv(input_file, names=names)

# Split the data into training and testing sets
# Separando x e y
X = df.drop('Resultado', axis = 1)
y = df['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Balanceamento de classe
oversample = SMOTE()
X_train_b, y_train_b = oversample.fit_resample(X_train,y_train)

# Implement a neural network
clf_mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=1000)
clf_mlp.fit(X_train_b, y_train_b)
y_pred_mlp = clf_mlp.predict(X_test)

# Implement Decision Tree classifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train_b, y_train_b)
y_pred_dt = clf_dt.predict(X_test)

# Implement K-Nearest Neighbors classifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train_b, y_train_b)
y_pred_knn = clf_knn.predict(X_test)

# Implement Support Vector Machine classifier
clf_svm = SVC()
clf_svm.fit(X_train_b, y_train_b)
y_pred_svm = clf_svm.predict(X_test)

print('Accuracy Scores:')
print('Neural Network:', accuracy_score(y_test, y_pred_mlp))
print('Decision Tree:', accuracy_score(y_test, y_pred_dt))
print('K-Nearest Neighbors:', accuracy_score(y_test, y_pred_knn))
print('Support Vector Machine:', accuracy_score(y_test, y_pred_svm))
print('\n')

print('F1 Scores:')
print('Neural Network:', f1_score(y_test, y_pred_mlp))
print('Decision Tree:', f1_score(y_test, y_pred_dt))
print('K-Nearest Neighbors:', f1_score(y_test, y_pred_knn))
print('Support Vector Machine:', f1_score(y_test, y_pred_svm))
print('\n')

print('Confusion Matrix:')
print('Neural Network:')
print(confusion_matrix(y_test, y_pred_mlp))
print('Decision Tree:')
print(confusion_matrix(y_test, y_pred_dt))
print('K-Nearest Neighbors:')
print(confusion_matrix(y_test, y_pred_knn))
print('Support Vector Machine:')
print(confusion_matrix(y_test, y_pred_svm))
print('\n')

print('Classification Report:')
print('Neural Network:')
print(classification_report(y_test, y_pred_mlp))
print('Decision Tree:')
print(classification_report(y_test, y_pred_dt))
print('K-Nearest Neighbors:')
print(classification_report(y_test, y_pred_knn))
print('Support Vector Machine:')
print(classification_report(y_test, y_pred_svm))
