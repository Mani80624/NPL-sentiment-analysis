from Models.navie_bayes import NavieBayes
from NLP.normalizacion import TextNormalizer
import pandas as pd

# Cargar los objetos a utilizar
cleaner = TextNormalizer()
navieB = NavieBayes()

data = {
    "text": [
        # HIGH RISK (2)
        "I don't want to live anymore",
        "I feel like ending everything",
        "There is no reason to keep going",
        "I am tired of life",
        "I can't take this pain anymore",
        "I want to disappear forever",
        "Nothing matters anymore",
        "I feel completely hopeless",
        "I wish I could just vanish",
        "I am done with everything",
        "Life is meaningless",
        "I feel empty inside",
        "I don't see a future for myself",
        "I just want to sleep forever",
        "Everything is too much for me",

        # MEDIUM RISK (1)
        "I feel very sad and alone",
        "Lately I feel down",
        "I am not okay",
        "I feel overwhelmed with everything",
        "I am struggling to keep going",
        "I feel lost",
        "Nothing makes me happy anymore",
        "I feel tired all the time",
        "I don't enjoy things like before",
        "I feel disconnected from everyone",
        "I feel anxious and worried",
        "Sometimes I feel like giving up",
        "I feel emotionally drained",
        "I don't feel like myself",
        "I am going through a hard time",

        # LOW RISK (0)
        "Today was a great day",
        "I feel happy and calm",
        "I am enjoying my time with friends",
        "I feel motivated to work",
        "Life is good right now",
        "I am grateful for my family",
        "I feel relaxed",
        "I am excited about my future",
        "I feel content with my life",
        "Everything is going well",
        "I feel positive today",
        "I am in a good mood",
        "I feel satisfied with my progress",
        "I am feeling great",
        "I feel at peace"
    ],
    "label": [
        *([2]*15),
        *([1]*15),
        *([0]*15)
    ]
}

df = pd.DataFrame(data)
# Normalización del modelo
df["text"] = df["text"].apply(cleaner.clean_text)
# Entrenamiento del modelo
navieB.trainModel(df)

# Evaluación de modelo
navieB.evaluate()

# Clasificación de nuevos textos

test_hard = [
    "I just want everything to stop",
    "I am fine, just a little tired of life",
    "I don't care about anything anymore",
    "I feel numb most of the time",
    "I wish I could sleep and not wake up",
    "I am okay, just overwhelmed"
]

pred, prob = navieB.new_prediction(test_hard)
print(f'La predicción es: {pred}')

# Guardar el modelo
navieB.save_model("prueba_1NB")