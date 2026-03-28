from Models.svm_model import SVM
from NLP.normalizacion import TextNormalizer
import pandas as pd

# Cargar los objetos a utilizar
normalizado = TextNormalizer()
svm_model = SVM()

# =========================================================
# CARGA DEL DATASET
# =========================================================
# Se carga el dataset grande con las emociones de Plutchik.
# Debe contener dos columnas:
# - text
# - label
df = pd.read_csv("data/plutchik_dataset.csv")

# =========================================================
# NORMALIZACIÓN DEL TEXTO
# =========================================================
# Se reutiliza el normalizador del proyecto para mantener
# consistencia con la estructura del equipo.
df["text"] = df["text"].apply(normalizado.clean_text)

# =========================================================
# ENTRENAMIENTO DEL MODELO
# =========================================================
svm_model.enntrenamienoModelo(df)

# =========================================================
# EVALUACIÓN DEL MODELO
# =========================================================
svm_model.evaluacion()

# =========================================================
# CLASIFICACIÓN DE NUEVOS TEXTOS
# =========================================================
text_new = [
    "I feel happy and excited today",
    "I am lonely and broken inside",
    "I am afraid of the future",
    "I hate this situation so much",
    "I trust my family and feel safe",
    "I am surprised by the sudden news",
    "I am waiting for something important tomorrow"
]

# Normalización de textos nuevos
text_new = [normalizado.clean_text(text) for text in text_new]

pred, prob = svm_model.prediccion(text_new)

print(f"La predicción es: {pred}")
print(f"La probabilidad es: \n{prob}")

# =========================================================
# GUARDAR MODELO
# =========================================================
svm_model.guardar_modelo("Prueba_1SVM")