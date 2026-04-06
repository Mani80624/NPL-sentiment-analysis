class TextPreprocessor:
    def __init__(self, stopwords):
        self.stopwords = set(stopwords)

        # NO eliminar negaciones
        self.stopwords.discard("not")
        self.stopwords.discard("no")
        self.stopwords.discard("never")

    def get_stopwords(self):
        return list(self.stopwords)