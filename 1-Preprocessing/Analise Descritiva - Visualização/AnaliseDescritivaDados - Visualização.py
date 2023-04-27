import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/diabetesClearGrafico.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','idade','Resultado'] 
    target = 'Resultado'
    atributo = 'Número Gestações'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas  


    df_dispersao =df[[target, atributo]]
    df_dispersao = df_dispersao.groupby([target,atributo]).size().reset_index(name="tamanho")
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.5,1.5])

    ax = sns.scatterplot(data=df_dispersao, x=target, y = atributo, size="tamanho", legend=False, sizes=(1000,10000))
    plt.show()

if __name__ == "__main__":
    main()