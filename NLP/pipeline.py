from NLP.tokenizer import Tokenizer
from NLP.normalizacion import TextNormalizer
from NLP.stopwords import StopWordsRemover
from NLP.sentiment_analyzer import SentimentAnalyzer

class SentimentPipeline:
    """
    Flujo de trabajo del preprocesamiento de texto y analisis de emoción 
    dominante mediante Kmeans
    """
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        self.stopwords = StopWordsRemover()
        self.analyzer = SentimentAnalyzer()

    def process(self, text):

        clean,_ = self.normalizer.normalize(text)

        tokens = self.tokenizer.tokenize(clean)
        tokens_stopwords = self.stopwords.remove(tokens)

        emotion, counter = self.analyzer.analyze(tokens_stopwords)

        return {
            "clean_text": clean,
            "tokens": tokens,
            "tokens_stopwords": tokens_stopwords,
            "emotion": emotion,
            "counter": counter
        }
