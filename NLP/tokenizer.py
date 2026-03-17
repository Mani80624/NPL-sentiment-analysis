from nltk.tokenize import word_tokenize

class Tokenizer:

    def tokenize(self, text):

        tokens = word_tokenize(text)

        return tokens