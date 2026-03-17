from NLP.normalizacion import TextNormalizer
from NLP.tokenizer import Tokenizer
from NLP.stopwords import StopWordsRemover
from NLP.sentiment_analyzer import SentimentAnalyzer

class SentimentPipeline:

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        self.stopwords = StopWordsRemover()
        self.analyzer = SentimentAnalyzer()

    def process(self, text):

        clean, sentences = self.normalizer.normalize(text)

        tokens = self.tokenizer.tokenize(clean)
        tokens = self.stopwords.remove(tokens)

        emotion, counter = self.analyzer.analyze(tokens)

        return {
            "clean_text": clean,
            "sentences": sentences,
            "tokens": tokens,
            "emotion": emotion,
            "counter": counter
        }