
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
dados = pd.read_csv('dados_diabetes_normalizados.csv')
# Obtendo os nomes das colunas.
cols = list(dados.columns)
# Removendo colunas que nao serao inclusas
# na reducao de dimensionalidade.
cols.remove('Resultado')
# Instanciando um PCA. O parametro n_components
# indica a quantidade de dimensoes que a base
# original sera reduzida.
pca = PCA(n_components=2, whiten=True)
# Aplicando o pca na base breast_cancer.
# O atributo 'values' retorna um numpy.array
# de duas dimensões (matriz) contendo apenas
# os valores numericos do DataFrame.
dados_pca = pca.fit_transform(dados[cols].values)
# O metodo fit_transform retorna outro numpy.array
# de dimensao numero_objetos x n_components.
# Apos isso, instancia-se um novo DataFrame contendo
# a base de dados original com dimensionalidade
# reduzida.
dados_pca = pd.DataFrame(dados_pca,
columns=['comp1', 'comp2'])
