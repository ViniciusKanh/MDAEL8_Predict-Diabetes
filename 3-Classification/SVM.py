# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

def load_dataset(dataset='diabetes'):        
    if dataset == 'iris':
        # Load iris data and store in dataframe
        iris = datasets.load_iris()
        names = iris.target_names
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
    elif dataset == 'diabetes':
        # Load cancer data and store in dataframe
        diabetes = datasets.load_diabetes()
        names = diabetes.target_names
        df = pd.DataFrame(data=diabetes.names, columns=diabetes.feature_names)
        df['target'] = diabetes.target
    
    print(df.head())
    return names, df


def main():
    #load dataset
    input_file = '0-Datasets/diabetesClear.data'
    names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado']
    features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade']
    target = 'Resultado'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                    names = names) # Nome das colunas
    # Separating out the features
    target_names = ['Não Diabetico','Diabetico']
    X = df.loc[:, features].values

    df['target'] = target

    #X = df.drop('target', axis=1)
    #y = df.target.values

    # Separating out the target
    y = df.loc[:,target]

    print("Total samples: {}".format(X.shape[0]))
    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))
     

    # Scale the X data using Z-score
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    svm = SVC(kernel='poly', C=1) # poly, rbf, linear
    # training using train dataset
    svm.fit(X_train, y_train)
    # get support vectors
    print(svm.support_vectors_)
    # get indices of support vectors
    print(svm.support_)
    # get number of support vectors for each class
    print("Qtd Support vectors: ")
    print(svm.n_support_)
    # predict using test dataset
    y_hat_test = svm.predict(X_test)

     # Get test accuracy score

    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}%".format(f1))
    # Get test confusion matrix    

    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - SVM sklearn")      
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - SVM sklearn normalized" )  
    plt.show()

if __name__ == "__main__":
    main()