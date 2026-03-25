import re
import joblib


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#\w+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúñ']", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


modelo = joblib.load("modelo_svm_emociones.pkl")

textos_prueba = [
    "I feel happy and excited today",
    "I am lonely and broken inside",
    "I am afraid of the future",
    "I hate this situation so much",
    "I trust my family and feel safe",
    "I am surprised by the sudden news",
    "I am waiting for something important tomorrow"
]

print("===== PRUEBAS DEL MODELO SVM =====")

for texto in textos_prueba:
    texto_limpio = limpiar_texto(texto)
    prediccion = modelo.predict([texto_limpio])[0]
    probabilidades = modelo.predict_proba([texto_limpio])[0]
    clases = modelo.named_steps["svm"].classes_

    print("\nTexto:", texto)
    print("Texto limpio:", texto_limpio)
    print("Predicción:", prediccion)
    print("Probabilidades:")
    for clase, prob in zip(clases, probabilidades):
        print(f"  {clase}: {round(prob * 100, 2)}%")