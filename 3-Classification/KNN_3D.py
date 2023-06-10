import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Dados de exemplo
X = np.array([[1, 2], [2, 1], [2, 3], [4, 2], [3, 3], [3, 6], [4, 4], [6, 4], [5, 6], [7, 4]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Instanciando o classificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Ajustando o modelo aos dados
knn.fit(X, y)

# Pontos para predição
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Plotando os resultados
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Classificação utilizando KNN')
plt.show()
