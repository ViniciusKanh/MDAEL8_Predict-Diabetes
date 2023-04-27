import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importando os dados do Excel
input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
target = 'Resultado'
data = pd.read_csv(input_file,         # Nome do arquivo com dados #df =  data framing
                     names = names,      # Nome das colunas 
                     usecols = features,)  # Define as colunas que serão  utilizadas
     

# Selecionando as colunas com as variáveis
X = data.iloc[:, 0:8].values

# Cálculo do número de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotando o gráfico do cotovelo
plt.plot(range(1, 11), wcss)
plt.title('Método do cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

# Executando o algoritmo K-means com o número de clusters escolhido
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

# Plotando o gráfico de dispersão dos clusters
plt.scatter(X[:,0], X[:,1], c=pred_y)
plt.title('Distribuição dos clusters')
plt.xlabel('Número de gestações')
plt.ylabel('Nível de glicose')
plt.show()