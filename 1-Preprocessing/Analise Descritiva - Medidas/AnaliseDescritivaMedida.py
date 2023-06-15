import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/diabetesClear.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
    target = 'Glucose'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
    
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

    print("\n-----------------------------------------\n")
    print("\n-----------------------------------------\n")
    print(f"\nAnalise Descritiva Medidas da Coluna: {target}\n")

    print("\n-----------------------------------------\n")
    print(f"\nMedidas de tendência central da Coluna: {target}\n")
    print("Média: {:.2f}".format(media))
    print("Mediana: {:.2f}".format(mediana))
    print("Moda : {:.2f}".format(moda))
    print("\n-----------------------------------------\n")
    print(f"\nMedidas de dispersão da Coluna da Coluna: {target}\n")
    print("Variância : {:.2f}".format(variancia))
    print("Desvio padrão : {:.2f}".format(desvio_padrao))
    print("Coeficiente de variação (%): {:.2f}".format(coeficiente_variacao))
    print("Amplitude: {:.2f}".format(amplitude))
    print("\n-----------------------------------------\n")
    print(f"\nMedidas de posição relativa da Coluna: {target}\n")
    print('Z Score:\n{}\n'.format((df[target] - df[target].mean())/df[target].std())) # Z Score
    print("Primeiro quartil: {:.2f}".format(Q1))
    print("Segundo quartil (Mediana): {:.2f}".format(Q2))
    print("Terceiro quartil: {:.2f}".format(Q3))
    print("\n-----------------------------------------\n")
    print("\n-----------------------------------------\n")
    print(f"\nMedidas de associação da Coluna: {target}\n")
    print('Covariância: \n{}\n'.format(df.cov())) # Covariância
    print('\nCorrelação: \n{}'.format(df.corr())) # Correlação
    print("\n-----------------------------------------\n")
 
    # Plota o histograma dos valores da coluna Glucose
    #plt.hist(df['Glucose'], bins=20)
    #plt.title('Distribuição de Glucose')
    #plt.xlabel('Glucose')
    #plt.ylabel('Frequência')
    #plt.show()

       # Boxplot dos quartis
    plt.boxplot([df[target]])
    plt.title('Quartis da coluna {}'.format(target))
    plt.ylabel('Idade')
    plt.show()





if __name__ == '__main__':
    main()
