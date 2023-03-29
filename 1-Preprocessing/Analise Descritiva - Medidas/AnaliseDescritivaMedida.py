import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/diabetesClear.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
    target = 'Resultado'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
    
    # Medidas de tendência central
    media_Glucose = df['Glucose'].mean()
    mediana_Glucose = df['Glucose'].median()
    moda_Glucose = df['Glucose'].mode()[0]

    # Medidas de dispersão
    variancia_Glucose = df['Glucose'].var()
    desvio_padrao_Glucose = df['Glucose'].std()
    coeficiente_variacao_Glucose = (desvio_padrao_Glucose / media_Glucose) * 100
    amplitude_Glucose = df['Glucose'].max() - df['Glucose'].min()

    # Medidas de posição relativa
    Q1_Glucose = df['Glucose'].quantile(0.25)
    Q2_Glucose = df['Glucose'].quantile(0.5)
    Q3_Glucose = df['Glucose'].quantile(0.75)

    print("\n-----------------------------------------\n")
    print("Média de Glucose: {:.2f}".format(media_Glucose))
    print("Mediana de Glucose: {:.2f}".format(mediana_Glucose))
    print("Moda de Glucose: {:.2f}".format(moda_Glucose))
    print("\n-----------------------------------------\n")
    print("Variância de Glucose: {:.2f}".format(variancia_Glucose))
    print("Desvio padrão de Glucose: {:.2f}".format(desvio_padrao_Glucose))
    print("Coeficiente de variação de Glucose (%): {:.2f}".format(coeficiente_variacao_Glucose))
    print("Amplitude de Glucose: {:.2f}".format(amplitude_Glucose))
    print("\n-----------------------------------------\n")
    print('Z Score:\n{}\n'.format((df['Glucose'] - df['Glucose'].mean())/df['Glucose'].std())) # Z Score
    print("Primeiro quartil de Glucose: {:.2f}".format(Q1_Glucose))
    print("Segundo quartil (Mediana) de Glucose: {:.2f}".format(Q2_Glucose))
    print("Terceiro quartil de Glucose: {:.2f}".format(Q3_Glucose))
    print("\n-----------------------------------------\n")
    
    print("\nMedidas de Associação\n")
    print('Covariância: \n{}\n'.format(df.cov())) # Covariância
    print('\nCorrelação: \n{}'.format(df.corr())) # Correlação
 
    # Plota o histograma dos valores da coluna Glucose
    #plt.hist(df['Glucose'], bins=20)
    #plt.title('Distribuição de Glucose')
    #plt.xlabel('Glucose')
    #plt.ylabel('Frequência')
    #plt.show()

if __name__ == '__main__':
    main()
