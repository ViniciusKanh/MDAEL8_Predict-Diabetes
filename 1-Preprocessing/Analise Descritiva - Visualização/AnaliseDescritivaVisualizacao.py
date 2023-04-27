import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/diabetesClear.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
    features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
    target = 'Resultado'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas   

    # Criando uma nova coluna 'idade_categoria'
    binsIdade = [12, 17, 50, 90, 120] # limites das categorias
    labelsIdade = ['Adolescente', 'Adulto', 'Idoso', 'Ancião'] # rótulos das categorias
    df['idade_categoria'] = pd.cut(df['idade'], bins=binsIdade, labels=labelsIdade, ordered=False)

    # Criando uma nova coluna 'imc_categoria'
    binsImc = [0, 18.5, 25, 30, 35, 40] # limites das categorias
    labelsImc = ['Abaixo do Peso', 'Normal', 'Pré-obesidade', 'Obesidade Classe I', 'Obesidade Classe II'] # rótulos das categorias
    df['imc_categoria'] = pd.cut(df['IMC'], bins=binsImc, labels=labelsImc)

    binsInsulina = [0, 140, 199, 200] # limites das categorias
    labelsInsulina = ['Saudável', 'Resistência à Insulina', 'Possível Diabético'] # rótulos das categorias
    df['insulina_categoria'] = pd.cut(df['Insulina'], bins=binsInsulina, labels=labelsInsulina)

    for coluna in df.columns:
        # Histograma
        plt.figure()
        if coluna == 'idade':
            sns.histplot(data=df, x='idade_categoria', hue='Resultado', multiple='stack')
            plt.title(f'Histograma da coluna {coluna}')
        elif coluna == 'IMC':
            sns.histplot(data=df, x='imc_categoria', hue='Resultado', multiple='stack')
            plt.title(f'Histograma da coluna {coluna}')
        elif coluna == 'Insulina':
            sns.histplot(data=df, x='Insulina', hue='Resultado', multiple='stack')
            plt.title(f'Histograma da coluna {coluna}')
        else:
            sns.histplot(data=df, x=coluna, hue='Resultado', multiple='stack')
            plt.title(f'Histograma da coluna {coluna}')
        plt.savefig(f'1-Preprocessing/Graficos/Histograma/histograma_{coluna}.png')
        plt.close()

        # Gráfico de setores
        plt.figure()
        if coluna == 'idade':
            df.groupby(['idade_categoria', 'Resultado']).size().unstack().plot(kind='pie', subplots=True, legend=False)
            plt.title(f'Gráfico de setores da coluna {coluna}')
        elif coluna == 'IMC':
            df.groupby(['imc_categoria', 'Resultado']).size().unstack().plot(kind='pie', subplots=True, legend=False)
            plt.title(f'Gráfico de setores da coluna {coluna}')
        elif coluna == 'Insulina':
            df['insulina_categoria'].value_counts().plot(kind='pie')
            plt.title(f'Gráfico de setores da coluna {coluna}')
        else:
            df.groupby([coluna, 'Resultado']).size().unstack().plot(kind='pie', subplots=True, legend=False)
            plt.title(f'Gráfico de setores da coluna {coluna}')
        plt.savefig(f'1-Preprocessing/Graficos/Grafico de Setores/grafico_setores_{coluna}.png')
        plt.close()

        # Dispersão  
        plt.figure()
        sns.scatterplot(data=df, x=coluna, y='Resultado', hue='insulina_categoria')
        plt.title(f'Dispersão da coluna {coluna} em relação à coluna alvo')
        plt.savefig(f'1-Preprocessing/Graficos/Dispersao/dispersao_{coluna}.png')
        plt.close()

if __name__ == "__main__":
    main()