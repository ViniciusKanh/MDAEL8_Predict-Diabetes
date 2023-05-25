import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade'] 
target = 'Resultado'

input_file = '0-Datasets/diabetesClear.data' 

df = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names)

df.info()

dims = (12, 9)
fig, ax = plt.subplots(figsize=dims)
sns.histplot(data=df, x='Idade', hue='Resultado', bins='auto', element='step')
plt.legend(labels=['Diabetes', 'No Diabetes']);

plt.show()



