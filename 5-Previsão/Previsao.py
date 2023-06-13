import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import io

# Carrega o dataset
input_file = '0-Datasets/diabetesClear.data'
names = ['Número Gestações', 'Glucose', 'Pressão Arterial', 'Espessura da Pele', 'Insulina', 'IMC', 'Função Pedigree Diabetes', 'Idade', 'Resultado']
features = ['Número Gestações', 'Glucose', 'Pressão Arterial', 'Espessura da Pele', 'Insulina', 'IMC', 'Função Pedigree Diabetes', 'Idade']
target = 'Resultado'
df = pd.read_csv(input_file, names=names)

target_names = ['Não Diabético', 'Diabético']

# Divide o dataset em features (X) e target (y)
X = df[features]
y = df[target]

# Divide o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padroniza as features para terem média zero e variância unitária
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializa os classificadores
classifiers = [
    ("KNN", KNeighborsClassifier(n_neighbors=3)),
    ("SVM", SVC(kernel='rbf', C=1)),
    ("Árvore de Decisão", DecisionTreeClassifier(max_leaf_nodes=5)),
    ("RNA", MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42))
]

# Solicita a identificação do paciente
identificacao = input("Digite a identificação do paciente: ")

# Solicita os dados do paciente para fazer a previsão
print("Digite os dados do paciente:")
dados_coletados = []
dados = []
for feature in features:
    dado = float(input(f"{feature}: "))
    dados_coletados.append(dado)
    dados.append(dado)

# Padroniza os dados inseridos
dados = scaler.transform([dados])

# Realiza as previsões usando os classificadores
results = []
accuracies = []
for name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    previsao = classifier.predict(dados)
    results.append([name, target_names[previsao[0]]])
    accuracies.append(classifier.score(X_test, y_test))

# Criação do relatório em PDF
filename = os.path.join("5-Previsão", "Relatorios", f"Relatorio_{identificacao}.pdf")
doc = SimpleDocTemplate(filename, pagesize=letter)

elements = []

# Título
styles = getSampleStyleSheet()
title = Paragraph("Relatório Médico", styles['Title'])
elements.append(title)

# Informações do paciente
elements.append(Paragraph("Identificação do paciente:", styles['Heading2']))
elements.append(Paragraph(f"Identificação: {identificacao}", styles['Normal']))

# Informações do paciente
elements.append(Paragraph("Dados do paciente:", styles['Heading2']))

# Criação da tabela de informações do paciente
patient_data_table_data = [[f" Descrição ", f" Valor Coletado ", f" Valor Normalizado "]]
patient_data_values = [[f"{feature}", f"{dados_coletados[i]}", f"{dados[0][i]}"] for i, feature in enumerate(features)]

patient_data_table_data.extend(patient_data_values)

# Ajuste as larguras das colunas aqui
col_widths = [150, 100, 150]  # Defina os valores de largura desejados

patient_data_table = Table(patient_data_table_data, colWidths=col_widths)
patient_data_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))

elements.append(patient_data_table)

# Resultados dos classificadores em tabela
elements.append(Paragraph("Resultados dos Classificadores:", styles['Heading2']))
results_table_data = [["Classificador", "Resultado"]]
results_table_data.extend(results)
results_table = Table(results_table_data, hAlign="LEFT")
results_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                   ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                   ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                   ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                   ('FONTSIZE', (0, 0), (-1, 0), 12),
                                   ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                   ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                   ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
elements.append(results_table)

# Orientações médicas
elements.append(Paragraph("Orientações Médicas:", styles['Heading2']))
if previsao[0] == 0:
    elements.append(Paragraph("O paciente não apresenta sinais de diabetes.", styles['Normal']))
    # Acrescente aqui as orientações médicas para o caso do paciente não ser diabético.
else:
    elements.append(Paragraph("O paciente apresenta sinais de diabetes.", styles['Normal']))
    # Acrescente aqui as orientações médicas para o caso do paciente ser diabético.

# Caminho completo para salvar o arquivo
temp_directory = "./temp"
os.makedirs(temp_directory, exist_ok=True)
temp_filename = os.path.join(temp_directory, 'precision_plot.png')

# Gráfico de histograma
elements.append(Paragraph("Precisão dos Classificadores:", styles['Heading2']))
classifiers_names = [name for name, _ in classifiers]
precisions = [accuracy * 100 for accuracy in accuracies]
plt.bar(classifiers_names, precisions)
plt.xlabel("Classificador")
plt.ylabel("Precisão (%)")
plt.title("Precisão dos Classificadores")
plt.xticks(rotation=45)

# Salva o gráfico no arquivo temporário
plt.savefig(temp_filename)

# Adiciona o gráfico ao relatório
elements.append(Paragraph("Gráfico de Precisão dos Classificadores:", styles['Normal']))
elements.append(Image(temp_filename, width=500, height=300))

# ...

# Gera o PDF
doc.build(elements)

print("Relatório gerado com sucesso:", filename)