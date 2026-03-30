from Models.svm_model import SVM
import pandas as pd
import os

# =========================================================
# CARGAR MODELO
# =========================================================
svm_model = SVM()

# =========================================================
# CARGA DEL DATASET
# =========================================================
df = pd.read_csv("data/emotions_risk_scores_1.csv")

print("Valores faltantes por columna:")
print(df.isna().sum())

# Rellenar NaN con 0
df = df.fillna(0)

# =========================================================
# VALIDACIÓN CRUZADA 10-FOLD
# =========================================================
svm_model.validacion_cruzada(df, folds=10)

# =========================================================
# GRÁFICA DE MÉTRICAS POR FOLD
# =========================================================
svm_model.grafica_validacion_cruzada(df, folds=10, output_dir="results")

# =========================================================
# ENTRENAMIENTO DEL MODELO
# =========================================================
svm_model.entrenamientoModelo(df)

# =========================================================
# EVALUACIÓN DEL MODELO
# =========================================================
resultados = svm_model.evaluacion(output_dir="results")

print("\nAccuracy:")
print(resultados["accuracy"])

print("\nReporte de clasificación:")
print(resultados["report"])

print("\nMatriz de confusión:")
print(resultados["confusion_matrix"])

print("\nSe generaron archivos en la carpeta 'results':")
print("- svm_report.txt")
print("- svm_confusion_matrix.csv")
print("- svm_confusion_matrix.png")
print("- svm_cross_validation_per_fold.png")
print("- svm_cross_validation_per_fold.csv")

# =========================================================
# CLASIFICACIÓN DE NUEVOS REGISTROS
# =========================================================
samples = pd.DataFrame([
    {
        "anger_count": 1,
        "anticipation_count": 2,
        "disgust_count": 0,
        "fear_count": 1,
        "joy_count": 4,
        "sadness_count": 0,
        "surprise_count": 1,
        "trust_count": 3,
        "text_length": 20,
        "score": 8,
        "score_norm": 0.40,
        "score_wins": 0.40
    },
    {
        "anger_count": 5,
        "anticipation_count": 0,
        "disgust_count": 3,
        "fear_count": 4,
        "joy_count": 0,
        "sadness_count": 6,
        "surprise_count": 0,
        "trust_count": 0,
        "text_length": 18,
        "score": 22,
        "score_norm": 1.22,
        "score_wins": 1.22
    },
    {
        "anger_count": 2,
        "anticipation_count": 1,
        "disgust_count": 1,
        "fear_count": 2,
        "joy_count": 1,
        "sadness_count": 3,
        "surprise_count": 0,
        "trust_count": 1,
        "text_length": 22,
        "score": 12,
        "score_norm": 0.54,
        "score_wins": 0.54
    }
])

pred, prob = svm_model.prediccion(samples)

print("\nLa predicción es:")
print(pred)

print("\nLa probabilidad es:")
print(prob)

# =========================================================
# GUARDAR PREDICCIONES DE EJEMPLO EN CSV
# =========================================================
os.makedirs("results", exist_ok=True)

pred_df = samples.copy()
pred_df["prediccion_riesgo"] = pred

# Etiquetas en inglés para el archivo de salida
clases_en = ["Low", "Medium", "High"]

for i, clase in enumerate(clases_en):
    pred_df[f"prob_{clase}"] = prob[:, i]

pred_df.to_csv("results/svm_predicciones_ejemplo.csv", index=False, encoding="utf-8")

print("\nTambién se generó:")
print("- svm_predicciones_ejemplo.csv")

# =========================================================
# GUARDAR MODELO
# =========================================================
svm_model.guardar_modelo("Prueba_1SVM_riesgo")

print("\nModelo guardado como:")
print("- Prueba_1SVM_riesgo.pkl")