from nltk.corpus import stopwords

class StopWordsRemover:

    def __init__(self, language="english"):

        palabras_importantes = {
            "i","me","my","mine","myself",
            "you","your","we","us","they","them",
            "no","not","never","nothing","nor",
            "always","everything",
            "very","so","too","really",
            "but","because","although","however",
            "anymore","anyone","nobody","someone"
        }

        stop_words = set(stopwords.words(language))
        self.stop_words = stop_words - palabras_importantes

    def remove(self, tokens):

        filtered = [t for t in tokens if t.lower() not in self.stop_words]

        return filtered
