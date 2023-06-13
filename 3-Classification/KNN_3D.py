import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Carregar o conjunto de dados
input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC', 'Função Pedigree Diabete', 'Idade', 'Resultado']
features = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC', 'Função Pedigree Diabete', 'Idade']
target = 'Resultado'
df = pd.read_csv(input_file, names=names)

# Dados de exemplo
X = df[features].values
y = df[target].values

# Normalização Min-Max
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Instanciando o classificador KNN com distância euclidiana
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Ajustando o modelo aos dados normalizados
knn.fit(X_scaled, y)

# Pontos para predição
x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])

# Plotando os resultados em 3D
Z = Z.reshape(xx.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contourf(xx, yy, Z, alpha=0.4)
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, s=20, edgecolor='k')
ax.set_xlabel('Número de Gestação (normalizado)')
ax.set_ylabel('Glucose (normalizado)')
ax.set_zlabel('pressao Arterial (normalizado)')
ax.set_title('Classificação utilizando KNN com normalização Min-Max e distância euclidiana')

# Legenda para as classes
diabetico_patch = plt.scatter([], [], c='r', s=20, edgecolor='k')
nao_diabetico_patch = plt.scatter([], [], c='b', s=20, edgecolor='k')
ax.legend((diabetico_patch, nao_diabetico_patch), ('Diabético', 'Não Diabético'), loc='upper right')

# Legenda para a região de classificação
classificacao_patch = plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.4)
nao_classificacao_patch = plt.Rectangle((0, 0), 1, 1, fc='purple', alpha=0.4)
ax.legend((classificacao_patch, nao_classificacao_patch), ('Região Classificada como Não Diabético', 'Região Classificada como Diabético'), loc='lower left')

plt.show()
