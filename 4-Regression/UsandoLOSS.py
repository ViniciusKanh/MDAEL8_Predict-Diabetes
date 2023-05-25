import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load the data
input_file = '0-Datasets/diabetesClear.data' 
names = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC', 'Função Pedigree Diabete', 'Idade', 'Resultado']
df = pd.read_csv(input_file, names=names)

# Split data into input and output variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
