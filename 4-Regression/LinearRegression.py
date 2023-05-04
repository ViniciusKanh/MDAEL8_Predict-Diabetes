import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# carrega os dados
input_file = '0-Datasets/diabetesClear.data' 
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
df = pd.read_csv(input_file, names=names)

target_names = ['Não Diabetico', 'Diabetico']

# separa em set de treino e teste
X = df.drop('Resultado', axis=1)
y = df['Resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = LinearRegression()
regr.fit(X_train, y_train)

r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R2 no set de treino: %.2f' % r2_train)
print('R2 no set de teste: %.2f' % r2_test)

y_pred = regr.predict(X_test)
abs_error = mean_absolute_error(y_pred, y_test)
print('Erro absoluto no set de treino: %.2f' % abs_error)

