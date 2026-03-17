from nltk.tokenize import word_tokenize

class Tokenizer:
    """Tokeniza las palabras con la librería tokenize"""
    def tokenize(self, text):
        tokens = word_tokenize(text)

        return tokens