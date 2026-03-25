import re
import joblib


# =========================================================
# FUNCIÓN DE LIMPIEZA
# =========================================================
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#\w+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúñ']", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# =========================================================
# CARGAR MODELO
# =========================================================
MODEL_PATH = "modelo_svm_emociones.pkl"
modelo = joblib.load(MODEL_PATH)

print("Modelo cargado correctamente.")


# =========================================================
# PREDICCIÓN DE TEXTO NUEVO
# =========================================================
texto = input("Escribe un texto en inglés: ")
texto_limpio = limpiar_texto(texto)

prediccion = modelo.predict([texto_limpio])[0]
probabilidades = modelo.predict_proba([texto_limpio])[0]
clases = modelo.named_steps["svm"].classes_

print("\nTexto limpio:", texto_limpio)
print("Emoción predicha:", prediccion)

print("\nProbabilidades:")
for clase, prob in zip(clases, probabilidades):
    print(f"{clase}: {round(prob * 100, 2)}%")