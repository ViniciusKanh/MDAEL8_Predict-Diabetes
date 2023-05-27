import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações', 'Glucose', 'Pressão Arterial', 'Espessura da Pele', 'Insulina', 'IMC', 'Função Pedigree Diabete', 'Idade', 'Resultado']
df = pd.read_csv(input_file, names=names)

# Split the dataset into training and testing sets
X = df.drop('Resultado', axis=1)
y = df['Resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a neural network classifier with 2 hidden layers each of 4 neurons
nn = MLPClassifier(hidden_layer_sizes=(4, 4), max_iter=1000)

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

# Calculate the classification report for each classifier
classification_report_nn = classification_report(y_test, y_pred_nn, output_dict=True)
classification_report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
classification_report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
classification_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)

# Create a DataFrame with the classification reports
df_output = pd.DataFrame({
    'MLPClassifier': classification_report_nn['1'],
    'DecisionTreeClassifier': classification_report_dt['1'],
    'KNeighborsClassifier': classification_report_knn['1'],
    'SVM': classification_report_svm['1']
})

# Transpose the DataFrame
df_output = df_output.transpose()

# Save the DataFrame to Excel file
output_file = '0-Datasets/MelhoresClassificadores.xlsx'
df_output.to_excel(output_file)

print("Classification reports saved to", output_file)
