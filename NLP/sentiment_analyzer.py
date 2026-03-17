from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter


class SentimentAnalyzer:

    def __init__(self, k=3):

        self.k = k

        self.emotion_lexicon = {
            "sad": ["sad", "cry", "hopeless", "lonely"],
            "anger": ["hate", "angry", "rage"],
            "joy": ["happy", "love", "smile"],
            "fear": ["afraid", "scared", "panic"]
        }

    def map_emotion(self, word):

        for emotion, words in self.emotion_lexicon.items():
            if word in words:
                return emotion

        return "neutral"

    def analyze(self, tokens):

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
        dominant_cluster = cluster_counts.most_common(1)[0][0]

        # palabras del cluster dominante
        dominant_words = [
            tokens[i] for i in range(len(tokens))
            if labels[i] == dominant_cluster
        ]

        # mapear a emociones
        emotions = [self.map_emotion(w) for w in dominant_words]

        emotion_counter = Counter(emotions)

        dominant_emotion = emotion_counter.most_common(1)[0][0]

        return dominant_emotion, emotion_counter