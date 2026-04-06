import pandas as pd
import numpy as np

# 1. Definir la Matriz de Decisión con los datos extraídos de tus reportes
data = {
    'Modelo': ['Naive Bayes', 'Random Forest', 'CNN', 'SVM'],
    'Accuracy': [0.71, 0.74, 0.84, 0.99],
    'Precision': [0.71, 0.75, 0.85, 0.99],
    'Recall': [0.71, 0.74, 0.84, 0.99],
    'F1_Score': [0.71, 0.74, 0.84, 0.99]
}

df = pd.DataFrame(data)

# 2. Definir los Pesos personalizados (deben sumar 1.0)
# Orden: [Accuracy, Precision, Recall, F1_Score]
weights = np.array([0.15, 0.20, 0.25, 0.40])

# --- ALGORITMO TOPSIS ---

# Extraer valores numéricos para el cálculo
matrix = df[['Accuracy', 'Precision', 'Recall', 'F1_Score']].values

# Normalización Vectorial
norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))

# Matriz Ponderada (Multiplicar por los pesos asignados)
weighted_matrix = norm_matrix * weights

# Determinar Soluciones Ideales
# Como todas son métricas de beneficio, el mejor es el máximo y el peor el mínimo
ideal_best = np.max(weighted_matrix, axis=0)
ideal_worst = np.min(weighted_matrix, axis=0)

# Cálculo de Distancias Euclidianas
dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

# Puntaje de Proximidad Relativa (Pi)
df['Puntaje_TOPSIS'] = dist_worst / (dist_best + dist_worst)

# Generar el Ranking Final
df['Ranking'] = df['Puntaje_TOPSIS'].rank(ascending=False).astype(int)

# Mostrar resultados finales ordenados por importancia
print(df.sort_values(by='Ranking').reset_index(drop=True))