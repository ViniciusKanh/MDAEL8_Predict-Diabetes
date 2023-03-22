import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        diabetes = pd.read_csv('/kaggle/input/predict-diabities/diabetes.csv')
        diabetes.info()
        diabetes.shape
        diabetes.head()
# To check the entries for each variable
        for col in diabetes.columns:
            print('{} : {}'.format(col, diabetes[col].unique()))

     # To correct values of 0 to NAN
        replace_value = ['Glucose', 'BloodPressure',
                         'SkinThickness', 'Insulin', 'BMI']

    for col in replace_value:
        diabetes[col].replace({0: np.nan}, inplace=True)

# To visualize the missing values / NAN
        plt.figure(figsize=(20, 10))
        sns.heatmap(diabetes.isnull(), cbar=False, cmap=plt.cm.PuBu)

        # To check the percentage of missing value / NAN
        df = pd.DataFrame()
        df['Column'] = diabetes.columns
        df['Missing_Count'] = [diabetes[col].isnull().sum()
                            for col in diabetes.columns]
        df['Percentage(%)'] = [round((diabetes[col].isnull().sum() /
                                    diabetes.shape[0])*100, 2) for col in diabetes.columns]

        df

        # To visualize data distribution for variables with missing values
    for col in replace_value:
        plt.figure()
        plt.tight_layout()
        sns.set(rc={'figure.figsize': (10,5)})

        f, ax = plt.subplots()
        plt.gca().set(xlabel=col, ylabel='Frequency')
        sns.histplot(diabetes[col], bins=10, kde=True)
