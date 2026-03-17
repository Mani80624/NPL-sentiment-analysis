class RiskFeatureExtractor:
    """
    Clase que especifica qué emociones estan relacionadas a qué palabras y lleva un conteo dependiendo 
    en que emoción se clasifique:
    hopeless, death o goodbye
    """

    def __init__(self):
        """
        Palabras asociadas a emociones
        """

        self.hopeless_words = {
            "hopeless", "worthless", "empty",
            "tired", "alone", "useless",
            "nothing", "nobody"
        }

        self.death_words = {
            "die", "death", "suicide",
            "kill", "disappear", "end"
        }

        self.goodbye_patterns = {
            "goodbye", "farewell", "last time"
        }

    def extract(self, tokens):

        features = {
            "hopeless_count": 0,
            "death_count": 0,
            "goodbye_count": 0,
            "text_length": len(tokens)
        }
        """Recorre la lista de tokens y verifica si las palabras se encuentran en los diccionarios
        del método constructor, en caso de que encuentre uno aumenta en el contador del diccionario"""
        for t in tokens:

            if t in self.hopeless_words:
                features["hopeless_count"] += 1

            if t in self.death_words:
                features["death_count"] += 1

            if t in self.goodbye_patterns:
                features["goodbye_count"] += 1

        return features