import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('0-Datasets/diabetesClear.data')


for coluna in df.columns:
    # Histograma
    plt.figure()
    sns.histplot(data=df, x=coluna)
    plt.title(f'Histograma da coluna {coluna}')
    plt.savefig(f'1-Preprocessing/Graficos/\Histograma/historgrama_{coluna}.png')
    plt.close()

    # Gráfico de setores
    plt.figure()
    df[coluna].value_counts().plot(kind='pie')
    plt.title(f'Gráfico de setores da coluna {coluna}')
    plt.savefig(f'1-Preprocessing/Graficos/Grafico de Setores/grafico_setores_{coluna}.png')
    plt.close()

    # Dispersão
    plt.figure()
    sns.scatterplot(data=df, x=coluna, y='Outcome')
    plt.title(f'Dispersão da coluna {coluna} em relação à coluna alvo')
    plt.savefig(f'1-Preprocessing/Graficos/Dispersao/dispersao_{coluna}.png')
    plt.close()
