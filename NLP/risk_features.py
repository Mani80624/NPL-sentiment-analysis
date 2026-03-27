class RiskFeatureExtractor:

    def __init__(self, lexicon_path):
        self.emotions = [
            "anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"
        ]

        # Diccionario: palabra -> emociones activas
        self.lexicon = {}

        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                word, emotion, value = line.strip().split("\t")

                if emotion not in self.emotions:
                    continue

                if int(value) == 1:
                    if word not in self.lexicon:
                        self.lexicon[word] = set()

                    self.lexicon[word].add(emotion)

    def extract(self, tokens):

        # Inicializar contador de emociones
        features = {f"{e}_count": 0 for e in self.emotions}
        features["text_length"] = len(tokens)

        for t in tokens:
            if t in self.lexicon:
                for emotion in self.lexicon[t]:
                    features[f"{emotion}_count"] += 1

        return features