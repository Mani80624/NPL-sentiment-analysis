from NLP.pipeline import SentimentPipeline
from NLP.risk_features import RiskFeatureExtractor
from NLP.risk_scorer import RiskScorer

class RiskDetectionPipeline:

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