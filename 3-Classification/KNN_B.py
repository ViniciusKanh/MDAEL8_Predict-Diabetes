# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN

# Calculate distance between two points
def minkowski_distance(a, b, p=2):
    # Store the number of dimensions
    dim = len(a)
    # Set initial distance to 0
    distance = 0

    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p

    distance = distance ** (1 / p)
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)

        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], index=y_train.index)

        # Sort distances, and only consider the k closest points

        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]

        # Append prediction to output list
        y_hat_test.append(prediction)

    return y_hat_test

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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
def main():
    
    names = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC',         'Função Pedigree Diabete', 'Idade', 'Resultado']
    features = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC',            'Função Pedigree Diabete', 'Idade']
    target = 'Resultado'

    input_file = '0-Datasets/diabetesClear.data'
    df = pd.read_csv(input_file, names=names)

    target_names = ['Não Diabetico', 'Diabetico']

    # Separating out the features
    X = df.loc[:, features].values

    df['target'] = target

    # X = df.drop('target', axis=1)
    # y = df.target.values

    # Separating out the target
    y = df.loc[:, target]

    print("Total samples: {}".format(X.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    oversample = ADASYN()
    X_train_b, y_train_b = oversample.fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # STEP 1 - TESTS USING knn classifier write from scratch
    # Make predictions on test dataset using knn classifier
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test,y_hat_test, average='macro')
    print("Accuracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}%".format(f1))
    
    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - K-NN")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - K-NN normalized")

    # STEP 2 - TESTS USING knn classifier from sk-learn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from sk-learn: {:.2f}%".format(f1))

    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - K-NN sklearn")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - K-NN sklearn normalized")

    # STEP 3 - TESTS USING knn classifier on balanced data using SMOTE
    X_train_B = scaler.transform(X_train_b)
    X_test_B = scaler.transform(X_test)

    knn_SMOTE = KNeighborsClassifier(n_neighbors=5)
    knn_SMOTE.fit(X_train_B, y_train_b)
    y_hat_test_B = knn_SMOTE.predict(X_test_B)

    # Get test accuracy score for balanced data using SMOTE
    accuracy_B = accuracy_score(y_test, y_hat_test_B) * 100
    f1_B = f1_score(y_test, y_hat_test_B, average='macro')
    print("Accuracy K-NN on balanced data using SMOTE: {:.2f}%".format(accuracy_B))
    print("F1 Score K-NN on balanced data using SMOTE: {:.2f}%".format(f1_B))

    # Get test confusion matrix for balanced data using SMOTE
    cm_B = confusion_matrix(y_test, y_hat_test_B)
    plot_confusion_matrix(cm_B, target_names, False, "Confusion Matrix - K-NN SMOTE")
    plot_confusion_matrix(cm_B, target_names, True, "Confusion Matrix - K-NN SMOTE normalized")
    
if __name__ == "__main__":
    main()

