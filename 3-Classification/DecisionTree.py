import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import pydotplus
import matplotlib.pyplot as plt

# Carregar a base de dados
names = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade','Resultado'] 
features = ['Número Gestações','Glucose','pressao Arterial','Expessura da Pele','Insulina','IMC','Função Pedigree Diabete','Idade'] 
target = 'Resultado'

input_file = '0-Datasets/diabetesClear.data' 
data = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names)      # Nome das colunas 

# Verificar se a base de dados é supervisionada ou não supervisionada
if target in data.columns:
    is_supervised = True
    target_values = data[target].unique()
    num_classes = len(target_values)
    class_counts = data[target].value_counts()

    print("A base de dados é supervisionada.")
    print("Número de classes:", num_classes)
    print("Classes:", target_values)
    print("Contagem de instâncias em cada classe:")
    print(class_counts)
else:
    is_supervised = False
    print("A base de dados é não supervisionada.")

 # Dividir o conjunto de dados entre atributos de entrada e rótulos/target
X = data[features]
y = data[target]

# Dividir o conjunto de dados entre treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de árvore de decisão
model = DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(X_train, y_train)

# Avaliar o modelo
accuracy = model.score(X_test, y_test)
print("Acurácia do modelo:", accuracy)

# Converter as classes para uma lista de strings
class_names = [str(class_name) for class_name in model.classes_]

# Criar um gráfico da árvore de decisão
dot_data = tree.export_graphviz(model, out_file=None, feature_names=features, class_names=class_names, filled=True, rounded=True)

# Converter o gráfico em um objeto graphviz
graph = pydotplus.graph_from_dot_data(dot_data)

# Exibir o gráfico
graphviz.Source(graph.to_string())

# Salvar o gráfico em um arquivo PDF
graph.write_pdf("3-Classification/Arvore de Decisão/arvore_decisao.pdf")   

# Função auxiliar para exibir os rótulos True e False nos caminhos
def plot_decision_tree_with_labels(model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(model,
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True,
                   rounded=True,
                   impurity=False,
                   ax=ax)
    text_objects = ax.findobj(plt.Text)
    for text in text_objects:
        text_str = text.get_text()
        if text_str.startswith('X'):
            text.set_text(text_str + ' = True')
        elif text_str.startswith('X') and '<=' in text_str:
            text.set_text(text_str.replace('X', ''))
        elif text_str.startswith('X') and '>' in text_str:
            text.set_text(text_str.replace('X', '') + ' = False')
    plt.show()

# Plotar o gráfico da árvore de decisão com os caminhos
plot_decision_tree_with_labels(model, features, class_names)