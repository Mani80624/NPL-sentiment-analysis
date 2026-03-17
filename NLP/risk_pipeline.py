from NLP.pipeline import SentimentPipeline
from NLP.risk_features import RiskFeatureExtractor
from NLP.risk_scorer import RiskScorer

class RiskDetectionPipeline:
    """
    Flujo de trabajo que hace match con los niveles de riesgo y la emoción 
    dominante, entrega un diccionario con los siguientes datos:
    clean text, tokens, tokens_stopwords, emotion, countoe, risk score, risk level y risk feature
    """
    def __init__(self):

        self.sentiment_pipeline = SentimentPipeline()
        self.feature_extractor = RiskFeatureExtractor()
        self.scorer = RiskScorer()

    def process(self, text):

        result = self.sentiment_pipeline.process(text)

        tokens = result["tokens"]

        features = self.feature_extractor.extract(tokens)

        score, level = self.scorer.score(features)

        result["risk_score"] = score
        result["risk_level"] = level
        result["risk_features"] = features

        return result