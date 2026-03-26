import pandas as pd
from Models.svm_model import SVM

# Cargar dataset
df = pd.read_csv("SVM/plutchik_dataset.csv")

# Crear modelo
modelo_svm = SVM()

# Entrenar
modelo_svm.enntrenamienoModelo(df)

# Evaluar
modelo_svm.evaluacion()

# Probar predicción
texto_prueba = ["I feel happy and excited today"]
pred, proba = modelo_svm.prediccion(texto_prueba)

print("\nPredicción:")
print(pred)

print("\nProbabilidades:")
print(proba)

# Guardar modelo
modelo_svm.guardar_modelo("modelo_svm_emociones")