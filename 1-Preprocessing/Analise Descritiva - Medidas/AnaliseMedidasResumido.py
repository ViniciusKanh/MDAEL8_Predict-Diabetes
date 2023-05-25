import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def display_table(headers, data):
    table = tabulate(data, headers=headers, tablefmt='grid')
    print(table)

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/diabetesClear.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
    target = 'IMC'
    df = pd.read_csv(input_file, names=names)
    
    # Medidas de tendência central
    media = df[target].mean()
    mediana = df[target].median()
    moda = df[target].mode()[0]

    # Medidas de dispersão
    variancia = df[target].var()
    desvio_padrao = df[target].std()
    coeficiente_variacao = (desvio_padrao / media) * 100
    amplitude = df[target].max() - df[target].min()

    # Medidas de posição relativa
    Q1 = df[target].quantile(0.25)
    Q2 = df[target].quantile(0.5)
    Q3 = df[target].quantile(0.75)

    # Exibir medidas de tendência central e dispersão em tabela
    headers = ['Medida', 'Valor']
    data = [
        ['Média', "{:.2f}".format(media)],
        ['Mediana', "{:.2f}".format(mediana)],
        ['Moda', "{:.2f}".format(moda)],
        ['Variância', "{:.2f}".format(variancia)],
        ['Desvio padrão', "{:.2f}".format(desvio_padrao)],
        ['Coeficiente de variação (%)', "{:.2f}".format(coeficiente_variacao)],
        ['Amplitude', "{:.2f}".format(amplitude)]
    ]
    print("\nMedidas de tendência central e dispersão da Coluna: {}\n".format(target))
    display_table(headers, data)

    # Exibir medidas de posição relativa
    print("\nMedidas de posição relativa da Coluna: {}\n".format(target))
    print("Primeiro quartil: {:.2f}".format(Q1))
    print("Segundo quartil (Mediana): {:.2f}".format(Q2))
    print("Terceiro quartil: {:.2f}".format(Q3))

    print("\n-----------------------------------------\n")
    print("\n-----------------------------------------\n")
    print(f"\nMedidas de associação da Coluna: {target}\n")
    print('Covariância: \n{}\n'.format(df.cov())) # Covariância
    print('\nCorrelação: \n{}'.format(df.corr())) # Correlação
    print("\n-----------------------------------------\n")

    # Boxplot dos quartis
    plt.boxplot([df[target]])
    plt.title('Quartis da coluna {}'.format(target))
    plt.ylabel('Idade')
    plt.show()

if __name__ == '__main__':
    main()
