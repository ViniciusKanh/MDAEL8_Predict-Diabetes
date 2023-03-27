import pandas as pd
from sklearn.decomposition import PCA

# Ler o arquivo csv com a base de dados normalizada
df_norm = pd.read_csv('dados_diabetes_normalizados.csv')

# Separar as variáveis dependentes da variável de saída
X_norm = df_norm.drop('Resultado', axis=1)
y_norm = df_norm['Resultado']

# Aplicar a PCA às variáveis dependentes normalizadas
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_norm)

# Criar um novo dataframe com as variáveis dependentes reduzidas e a variável de saída
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
df_pca['Resultado'] = y_norm

# Salvar a base de dados reduzida em um novo arquivo csv
df_pca.to_csv('dados_diabetes_reduzidos.csv', index=False)