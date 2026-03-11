# Natural Languaje Toolkit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download("stopwords")
#nltk.download("punkt_tab")

# Stopwords estándar de NLTK
stop_words = set(stopwords.words("english"))


# Palabras que NO queremos eliminar
def palabras():
    palabras_importantes = {
        "i","me","my","mine","myself",
        "you","your","we","us","they","them",
        "no","not","never","nothing","nor",
        "always","everything","nothing",
        "very","so","too","really",
        "but","because","although","however",
        "anymore","anyone","nobody","someone"
    }

    # Stopwords personalizadas
    custom_stopwords = stop_words - palabras_importantes
    return custom_stopwords

def stopWords(texto, stopwords_personalizadas):

    tokens = word_tokenize(texto.lower())
    tokens_filtrados = [w for w in tokens if w not in stopwords_personalizadas]
    return tokens_filtrados



limpio = stopWords("I went to the university today and met some friends. We talked about our classes and the projects we need to finish this week. It was a normal day and nothing unusual happened.",
                   palabras())

for i in limpio:
    print(i)
