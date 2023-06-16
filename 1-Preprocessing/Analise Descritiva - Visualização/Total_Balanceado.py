import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/diabetesClear.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
    features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
    target = 'Resultado'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas

    # Balanceamento do dataset usando SMOTE
    X = df[features].values
    y = df[target].values
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)
    df_balanced = pd.DataFrame(data=X_balanced, columns=features)
    df_balanced[target] = y_balanced

    label = ['Diabeticas','Não Diabeticas']
    cores = ['r','b']
    Diabeticas = df_balanced[target].value_counts()[1]
    NaoDiabeticas = df_balanced[target].value_counts()[0]
    total = Diabeticas + NaoDiabeticas
    y = np.array([Diabeticas, NaoDiabeticas])
    plt.pie(y , labels=label, colors=cores, autopct='%1.1f%%', startangle=90)
    plt.title('Total de Resultados')
    plt.show() 
    print("Total: {}\n".format(total) )
    print("Diabeticas: {:.2f}%\n".format((Diabeticas*100)/total))
    print("Não Diabeticas: {:.2f}%\n".format((NaoDiabeticas*100)/total))

if __name__ == "__main__":
    main()
