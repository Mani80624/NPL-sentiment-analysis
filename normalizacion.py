import re
from collections import Counter

# =============================
# Diccionario de emociones
# =============================
emociones = {
    "happy": "joy",
    "fun": "joy",
    "love": "joy",
    "blast": "joy",
    "good": "joy",

    "sad": "sadness",
    "late": "sadness",
    "wrong": "sadness",
    "lazy": "sadness",

    "angry": "anger",
    "nagging": "anger",
    "hassle": "anger",
    "crazy": "anger",

    "worried": "fear",
    "worry": "fear",
    "pressure": "fear",
    "problems": "fear",
    "challenging": "fear",
    "focus": "fear",
    "concentrate": "fear",

    "responsible": "responsibility",
    "school": "responsibility",
    "homework": "responsibility",
    "study": "responsibility",

    "freedom": "independence",
    "move": "independence",
    "explore": "independence",

    "parents": "family_conflict",
    "sister": "family_conflict",
    "home": "family_conflict"
}

negaciones = {"no","not","never","dont","don't","cant","can't"}
intensificadores = {"very","really","so","too","much","more"}
contrastes = {"but","however","although","though"}

# =============================
# Limpieza
# =============================
import re

def limpiar_texto(texto):

    # pasar a minúsculas
    texto = texto.lower()

    # eliminar URLs
    texto = re.sub(r'http\S+|www\S+', '', texto)

    # eliminar menciones
    texto = re.sub(r'@\w+', '', texto)

    # eliminar hashtags
    texto = re.sub(r'#\w+', '', texto)

    # eliminar signos de admiración y puntuación
    texto = re.sub(r'[^\w\sáéíóúñ]', ' ', texto)

    # eliminar espacios repetidos
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

# =============================
# Dividir en oraciones
# =============================
def dividir_oraciones(texto):

    oraciones = re.split(r'(?<=[\.\!\?])\s+', texto)

    return [o.strip() for o in oraciones if o.strip()]


# =============================
# Analizar oración
# =============================
def analizar_oracion(oracion):

    tokens = oracion.split()

    emociones_detectadas = []
    observaciones = []

    for i,palabra in enumerate(tokens):

        palabra_limpia = palabra.strip(".,!?")

        if palabra_limpia in emociones:

            emocion = emociones[palabra_limpia]

            if i>0 and tokens[i-1].strip(".,!?") in intensificadores:
                emocion = "alta_" + emocion

            if i>0 and tokens[i-1].strip(".,!?") in negaciones:
                emocion = "negada_" + emocion

            emociones_detectadas.append((palabra_limpia,emocion))

    if any(t.strip(".,!?") in contrastes for t in tokens):

        observaciones.append("contraste_detectado")

    return emociones_detectadas,observaciones


# =============================
# Analizar texto completo
# =============================
def analizar_texto(texto):

    texto_limpio = limpiar_texto(texto)

    oraciones = dividir_oraciones(texto_limpio)

    conteo = Counter()

    detalle = []

    for i,oracion in enumerate(oraciones):

        emociones_oracion,obs = analizar_oracion(oracion)

        for palabra,emocion in emociones_oracion:

            conteo[emocion]+=1

        detalle.append({
            "oracion":oracion,
            "emociones":emociones_oracion,
            "obs":obs
        })

    if len(conteo)>0:
        emocion_dominante = conteo.most_common(1)[0][0]
    else:
        emocion_dominante = "no_detectada"

    return texto_limpio,oraciones,conteo,emocion_dominante,detalle


# =============================
# PEDIR TEXTO AL USUARIO
# =============================
texto_usuario = input("Pega aquí tu texto largo:\n\n")


# =============================
# EJECUTAR ANÁLISIS
# =============================
texto_limpio,oraciones,conteo,emocion_dominante,detalle = analizar_texto(texto_usuario)


print("\n===== TEXTO LIMPIO =====")
print(texto_limpio)

print("\nTOTAL DE ORACIONES:",len(oraciones))

print("\nEMOCION DOMINANTE:")
print(emocion_dominante)

print("\nCONTEO DE EMOCIONES:")
print(conteo)

print("\n===== DETALLE POR ORACION =====")

for d in detalle:

    print("\nORACION:")
    print(d["oracion"])

    print("EMOCIONES:",d["emociones"])

    print("OBSERVACIONES:",d["obs"])