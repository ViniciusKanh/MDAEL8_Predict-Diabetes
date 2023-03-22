import pandas as pd
from sklearn.preprocessing import MinMaxScaler

    # Faz a leitura do arquivo
input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
target = 'Resultado'
df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas   
# Separar as variáveis dependentes da variável de saída
X = df.drop('Resultado', axis=1)
y = df['Resultado']

# Aplicar a normalização Min-Max às variáveis dependentes
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Criar um novo dataframe com as variáveis dependentes normalizadas e a variável de saída
df_norm = pd.DataFrame(X_norm, columns=X.columns)
df_norm['Resultado'] = y

# Salvar a base de dados normalizada em um novo arquivo csv
df_norm.to_csv('dados_diabetes_normalizados.csv', index=False)
