import re
import nltk

class TextNormalizer:
    """
    La clase limpia el texto, quita url y acentos
    """

    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_sentences(self, text):
        # opcional, no es necesario
        return nltk.sent_tokenize(text)

    def normalize(self, text):
        """Devuelve el texto limpio y segmentado"""
        clean = self.clean_text(text)
        sentences = self.split_sentences(clean)
        return clean, sentences