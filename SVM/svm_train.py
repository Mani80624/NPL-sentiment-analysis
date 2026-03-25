import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# FUNCIÓN DE LIMPIEZA
# =========================================================
# Realiza preprocesamiento básico:
# - minúsculas
# - elimina URLs
# - elimina menciones
# - elimina hashtags
# - elimina signos y emojis
# - elimina espacios repetidos
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#\w+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúñ']", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# =========================================================
# CARGA DEL DATASET
# =========================================================
# El dataset debe tener dos columnas:
# - text
# - label
DATASET_PATH = "plutchik_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("Primeras filas del dataset:")
print(df.head())

print("\nColumnas:")
print(df.columns)

print("\nCantidad por clase:")
print(df["label"].value_counts())


# =========================================================
# PREPROCESAMIENTO
# =========================================================
df["text"] = df["text"].apply(limpiar_texto)

X = df["text"]
y = df["label"]


# =========================================================
# DIVISIÓN DE DATOS
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nCantidad entrenamiento:", len(X_train))
print("Cantidad prueba:", len(X_test))


# =========================================================
# MODELO TF-IDF + SVM
# =========================================================
# TF-IDF realiza internamente tokenización y vectorización.
# SVM clasifica el texto en una emoción.
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("svm", SVC(
        kernel="linear",
        probability=True
    ))
])


# =========================================================
# ENTRENAMIENTO
# =========================================================
modelo.fit(X_train, y_train)
print("\nModelo entrenado correctamente.")


# =========================================================
# EVALUACIÓN
# =========================================================
y_pred = modelo.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


# =========================================================
# GUARDAR MODELO
# =========================================================
MODEL_PATH = "modelo_svm_emociones.pkl"
joblib.dump(modelo, MODEL_PATH)
print(f"\nModelo guardado en: {MODEL_PATH}")