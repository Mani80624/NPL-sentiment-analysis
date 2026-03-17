class RiskFeatureExtractor:

    def __init__(self):

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

        for t in tokens:

            if t in self.hopeless_words:
                features["hopeless_count"] += 1

            if t in self.death_words:
                features["death_count"] += 1

            if t in self.goodbye_patterns:
                features["goodbye_count"] += 1

        return features