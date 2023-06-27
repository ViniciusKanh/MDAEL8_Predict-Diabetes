from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import pickle
import json

app = Flask(__name__)

input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade']
target = 'Resultado'
df = pd.read_csv(input_file, names=names) 
X = df.loc[:, features].values
y = df.loc[:,target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Treinando o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Salvando o modelo
with open('API/modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)
with open('API/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Carregando o modelo
with open('API/modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)
with open('API/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(f"Dados recebidos: {data}")  # Imprimir os dados recebidos
    example = np.array(data['example']).reshape(1, -1)
    example = scaler.transform(example)  # Aplicar o mesmo escalonamento usado no treinamento
    prediction = modelo.predict(example)[0]
    print(f"Predição do modelo: {prediction}")  # Imprimir a previsão do modelo

    prediction = "Diabético" if prediction == 1 else "Não Diabético"
    print(f"Predição final: {prediction}")  # Imprimir a previsão final

    response = app.response_class(
        response=json.dumps({'prediction': prediction}, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)
