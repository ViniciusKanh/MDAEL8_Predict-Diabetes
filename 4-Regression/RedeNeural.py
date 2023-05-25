import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Load the dataset
input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
df = pd.read_csv(input_file, names=names)

# Split the dataset into training and testing sets
X = df.drop('Resultado', axis=1)
y = df['Resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a neural network classifier with 2 hidden layers each of 4 neurons
nn = MLPClassifier(hidden_layer_sizes=(4,4), max_iter=1000)

# Train the neural network classifier
nn.fit(X_train, y_train)

# Use the trained neural network classifier to predict results for the test set
y_pred_nn = nn.predict(X_test)

# Create and train decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Use trained decision tree classifier to make predictions on the test set
y_pred_dt = dt.predict(X_test)

# Create and train KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Use trained KNN classifier to make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Create and train SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Use trained SVM classifier to make predictions on the test set
y_pred_svm = svm.predict(X_test)

# Calculate the classification report and confusion matrix for each classifier
print("MLPClassifier classification report: ")
print(classification_report(y_test, y_pred_nn))
print(confusion_matrix(y_test, y_pred_nn))

print("DecisionTreeClassifier classification report: ")
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))

print("KNeighborsClassifier classification report: ") 
print(classification_report(y_test, y_pred_knn)) 
print(confusion_matrix(y_test, y_pred_knn))

print("SVM classification report: ") 
print(classification_report(y_test, y_pred_svm)) 
print(confusion_matrix(y_test, y_pred_svm))
