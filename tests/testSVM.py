from Models.svm_model import SVM
from NLP.normalizacion import TextNormalizer
import pandas as pd

# Cargar los objetos a utilizar
normalizado = TextNormalizer()
svm_model = SVM()

data = {
    "text": [
        # joy
        "I feel happy and full of energy today",
        "This moment makes me smile and enjoy life",
        "I am excited because everything is going well",
        "Today was fun and I feel amazing",
        "I feel joyful and calm this morning",

        # trust
        "I trust the people around me and feel safe",
        "I feel secure and supported by my family",
        "I know I can rely on my friends",
        "This place makes me feel safe and calm",
        "Their honesty makes me feel secure",

        # fear
        "I am afraid of what might happen tomorrow",
        "I feel anxious and nervous about the results",
        "I am worried and scared about the future",
        "This situation makes me nervous and afraid",
        "I feel terrified and unsure of what comes next",

        # surprise
        "I was surprised by the unexpected news",
        "That result shocked me completely",
        "I am amazed by what just happened",
        "The ending was shocking and unexpected",
        "I did not expect that at all",

        # sadness
        "I feel sad and empty today",
        "I am lonely and hurt inside",
        "This loss makes me feel deep sorrow",
        "I feel broken after what happened",
        "I want to cry because I feel so sad",

        # disgust
        "I feel disgust when I think about it",
        "That behavior is gross and disturbing",
        "I hate what I just saw",
        "This situation fills me with disgust",
        "That smell was disgusting and unbearable",

        # anger
        "I feel angry about what they did",
        "This injustice makes me furious",
        "I am mad and frustrated right now",
        "That lie made me angry and upset",
        "Their disrespect made me furious",

        # anticipation
        "I am waiting for something important to happen",
        "I feel excited about the future",
        "I expect good news very soon",
        "I am looking forward to tomorrow",
        "I feel anticipation before the event"
    ],
    "label": [
        *(["joy"] * 5),
        *(["trust"] * 5),
        *(["fear"] * 5),
        *(["surprise"] * 5),
        *(["sadness"] * 5),
        *(["disgust"] * 5),
        *(["anger"] * 5),
        *(["anticipation"] * 5)
    ]
}

df = pd.DataFrame(data)

# Normalización del modelo
df["text"] = df["text"].apply(normalizado.clean_text)

# Entrenamiento del modelo
svm_model.enntrenamienoModelo(df)

# Evaluación del modelo
svm_model.evaluacion()

# Clasificación de nuevos textos
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

# Guardar modelo
svm_model.guardar_modelo("Prueba_1SVM")