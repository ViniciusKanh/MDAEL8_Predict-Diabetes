import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lendo o arquivo csv
df = pd.read_csv('0-Datasets/diabetesClear.data')

# Criando uma nova coluna 'idade_categoria'
binsIdade = [12, 17, 50, 90, 120] # limites das categorias
labelsIdade = ['Adolescente', 'Adulto', 'Idoso', 'Ancião'] # rótulos das categorias
df['idade_categoria'] = pd.cut(df['idade'], bins=binsIdade, labels=labelsIdade)

# Criando uma nova coluna 'imc_categoria'
binsImc = [0, 18.5, 25, 30, 35, 40] # limites das categorias
labelsImc = ['Abaixo do Peso', 'Normal', 'Pré-obesidade', 'Obesidade Classe I', 'Obesidade Classe II'] # rótulos das categorias
df['imc_categoria'] = pd.cut(df['IMC'], bins=binsImc, labels=labelsImc)

# Loop pelas colunas
for coluna in df.columns:
    # Histograma
    plt.figure()
    if coluna == 'idade':
        sns.histplot(data=df, x='idade_categoria')
        plt.title(f'Histograma da coluna {coluna}')
    elif coluna == 'IMC':
        sns.histplot(data=df, x='imc_categoria')
        plt.title(f'Histograma da coluna {coluna}')
    else:
        sns.histplot(data=df, x=coluna)
        plt.title(f'Histograma da coluna {coluna}')
    plt.savefig(f'1-Preprocessing/Graficos/Histograma/histograma_{coluna}.png')
    plt.close()

    # Gráfico de setores
    plt.figure()
    if coluna == 'idade':
        df['idade_categoria'].value_counts().plot(kind='pie')
        plt.title(f'Gráfico de setores da coluna {coluna}')
    elif coluna == 'IMC':
        df['imc_categoria'].value_counts().plot(kind='pie')
        plt.title(f'Gráfico de setores da coluna {coluna}')
    else:
        df[coluna].value_counts().plot(kind='pie')
        plt.title(f'Gráfico de setores da coluna {coluna}')
    plt.savefig(f'1-Preprocessing/Graficos/Grafico de Setores/grafico_setores_{coluna}.png')
    plt.close()

    # Dispersão  
    plt.figure()
    sns.scatterplot(data=df, x=coluna, y='Resultado')
    plt.title(f'Dispersão da coluna {coluna} em relação à coluna alvo')
    plt.savefig(f'1-Preprocessing/Graficos/Dispersao/dispersao_{coluna}.png')
    plt.close()
