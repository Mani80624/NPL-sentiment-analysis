from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter


class SentimentAnalyzer:
    """
    Hace el analisis de sentimientos a partir de frecuencias y clusteres, en este caso solo son cuatro clusteres:
        {sad, anger, joy, fear}
    """
    def __init__(self, k=4):

        self.k = k # Clusteres

        self.emotion_lexicon = {
            "sad": ["sad", "cry", "hopeless", "lonely"],
            "anger": ["hate", "angry", "rage"],
            "joy": ["happy", "love", "smile"],
            "fear": ["afraid", "scared", "panic"]
        }

    def map_emotion(self, word):
        """Verificamos si las palabras se encuentran en alguna lista del diccionario de
        emotion_lexicon"""
        for emotion, words in self.emotion_lexicon.items():
            if word in words:
                return emotion

        return "neutral"

    def analyze(self, tokens):
        """
        Devuelve la emoción dominante y el conteo de emociones
        """
        if not tokens:
            return "neutral", {}

        # vectorizar
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(tokens)

        # clustering
        kmeans = KMeans(n_clusters=min(self.k, len(tokens)))
        labels = kmeans.fit_predict(X)

        # cluster dominante
        cluster_counts = Counter(labels)
        dominant_cluster = cluster_counts.most_common()[0][0]

        # palabras del cluster dominante
        dominant_words = [
            tokens[i] for i in range(len(tokens))
            if labels[i] == dominant_cluster
        ]
        
        # mapear a emociones
        emotions = [self.map_emotion(w) for w in dominant_words]
        
        emotion_counter = Counter(emotions)
        
        dominant_emotion = emotion_counter.most_common()[0][0]   
        return dominant_emotion, emotion_counter
