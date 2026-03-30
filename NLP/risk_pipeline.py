from NLP.pipeline import SentimentPipeline
from NLP.risk_features import RiskFeatureExtractor
from NLP.risk_scorer import RiskScorer


class RiskDetectionPipeline:
    """
    Flujo de trabajo completo para detección de riesgo.

    Output:
    - clean_text
    - tokens
    - emotion (del sentiment pipeline)
    - risk_score
    - risk_level
    - risk_features (conteo de emociones NRC)
    - dominant_emotion
    - dominant_count
    """

    def __init__(self):

        self.sentiment_pipeline = SentimentPipeline()

        # Cargar NRC Lexicon
        self.feature_extractor = RiskFeatureExtractor(
            "Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        )

        self.scorer = RiskScorer()

    def get_dominant_emotion(self, features):
        """
        Obtiene la emoción dominante basada en el conteo
        """

        emotion_counts = {
            k.replace("_count", ""): v
            for k, v in features.items()
            if "_count" in k
        }

        # Si todo está en 0, evitar bugs raros 
        if all(v == 0 for v in emotion_counts.values()):
            return "neutral", 0

        dominant = max(emotion_counts, key=emotion_counts.get)

        return dominant, emotion_counts[dominant]

    def process(self, text):
        """
        Ejecuta todo el pipeline
        """

        # 1. NLP base
        result = self.sentiment_pipeline.process(text)

        tokens = result["tokens"]

        # 2. Extraer emociones (NRC)
        features = self.feature_extractor.extract(tokens)

        # 3. Calcular riesgo
        score, level = self.scorer.score(features)

        # 4. Emoción dominante
        dominant_emotion, count = self.get_dominant_emotion(features)

        # 5. Agregar resultados
        result["risk_score"] = score
        result["risk_level"] = level
        result["risk_features"] = features
        result["dominant_emotion"] = dominant_emotion
        result["dominant_count"] = count

        return result