# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade']
target = 'Resultado'
df = pd.read_csv(input_file,    # Nome do arquivo com dados
                    names = names) #

print('The dataset contains '+str(df.shape[0])+' rows and '+str(df.shape[1])+' columns\n')
print('Preview of the dataset :')
df.head()

#renaming the columns to something easier 
df.columns=["Número Gestações","Glucose","pressao Arterial","Expessura da Pele","Insulina","IMC","Função Pedigree Diabete","Idade","Resultado"]

#checking distribution of target column 
df['Resultado'].value_counts()/(len(df))


#visualising outliers using Boxplots

plt.figure(figsize=(24,20))
x=1
for col in df.drop(['Resultado'],axis=1).columns:
    plt.subplot(4, 2, x)
    fig = df.boxplot(column=col)
    fig.set_title('')
    x=x+1

#visualising the distribution of the variables

plt.figure(figsize=(24,20))
x=1
for col in df.drop(['Resultado'],axis=1).columns:
    plt.subplot(4, 2, x)
    fig = df[col].hist(bins=20)
    fig.set_xlabel(col)
    x=x+1
    
    