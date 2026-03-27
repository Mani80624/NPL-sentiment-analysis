from collections import Counter
from os import path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer


class SentimentAnalyzer:
    """
    Analizador de emociones usando NRC Emotion Lexicon + KMeans clustering.
    """

    def __init__(self,
                 lexicon_path="Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                 k=3,
                 use_sentiment=False):

        self.k = k
        self.lemmatizer = WordNetLemmatizer()
        self.use_sentiment = use_sentiment

        self.emotion_lexicon = self.load_nrc_lexicon(lexicon_path)

        # IMPORTANTE: orden consistente
        self.emotions = sorted(self.emotion_lexicon.keys())

    def load_nrc_lexicon(self, path):
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["word", "emotion", "association"]

    # filtrar solo asociaciones válidas
        df = df[df["association"] == 1]

    # opcional: eliminar positive/negative
        if not self.use_sentiment:
            df = df[~df["emotion"].isin(["positive", "negative"])]

    # agrupar por emoción
        lexicon = df.groupby("emotion")["word"].apply(set).to_dict()

        return lexicon

    def normalize(self, word):
        return self.lemmatizer.lemmatize(word.lower())

    def word_to_vector(self, word):
        """
        Convierte una palabra a vector emocional binario.
        """
        word = self.normalize(word)

        vector = [
            1 if word in self.emotion_lexicon[emotion] else 0
            for emotion in self.emotions
        ]

        # normalización (opcional pero recomendable)
        vector = np.array(vector, dtype=float)
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)

        return vector

    def analyze(self, tokens):
        """
        Analiza una lista de tokens y devuelve:
        - emoción dominante
        - distribución de emociones
        """
        vectors = []

        for word in tokens:
            vec = self.word_to_vector(word)

            # ignorar palabras sin emoción
            if np.sum(vec) > 0:
                vectors.append(vec)

        if not vectors:
            return "neutral", {}

        X = np.array(vectors)

        # KMeans
        kmeans = KMeans(
            n_clusters=min(self.k, len(X)),
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        # usar centroides (mejor que solo conteo)
        centroids = kmeans.cluster_centers_
        dominant_cluster = np.argmax(np.sum(centroids, axis=1))

        # obtener vectores del cluster dominante
        dominant_vectors = X[labels == dominant_cluster]

        # sumar emociones
        emotion_scores = np.sum(dominant_vectors, axis=0)

        emotion_dict = {
            self.emotions[i]: round(float(emotion_scores[i]), 2)
            for i in range(len(self.emotions))
            if emotion_scores[i] > 0
        }

        if not emotion_dict:
            return "neutral", {}

        dominant_emotion = max(emotion_dict, key=emotion_dict.get)

        return dominant_emotion, emotion_dict