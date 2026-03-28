from NLP.pipeline import SentimentPipeline
from NLP.risk_features import RiskFeatureExtractor


class RiskDetectionPipeline:
    """
    Pipeline de extracción NLP + features emocionales

    Output:
    - clean_text
    - tokens
    - emotion
    - risk_features
    - dominant_emotion
    - dominant_count
    """

    def __init__(self):

        self.sentiment_pipeline = SentimentPipeline()

        self.feature_extractor = RiskFeatureExtractor(
            "Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        )

    def get_dominant_emotion(self, features):

        emotion_counts = {
            k.replace("_count", ""): v
            for k, v in features.items()
            if "_count" in k
        }

        if all(v == 0 for v in emotion_counts.values()):
            return "neutral", 0

        dominant = max(emotion_counts, key=emotion_counts.get)

        return dominant, emotion_counts[dominant]

    def process(self, text):

        result = self.sentiment_pipeline.process(text)

        tokens = result["tokens"]

        features = self.feature_extractor.extract(tokens)

        dominant_emotion, count = self.get_dominant_emotion(features)

        result["risk_features"] = features
        result["dominant_emotion"] = dominant_emotion
        result["dominant_count"] = count

        return result