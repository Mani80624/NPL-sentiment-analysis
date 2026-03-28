class ModelManager:

    def __init__(self):
        self.models = {
            1: self.naive_bayes,
            2: self.cnn,
            3: self.random_forest,
            4: self.svm
        }

    def predict(self, model_id, features):

        if model_id not in self.models:
            raise ValueError("Modelo no válido")

        return self.models[model_id](features)

    # -------- MODELOS --------

    def naive_bayes(self, features):
        score = sum(features.values())
        return score, self._nivel(score)

    def cnn(self, features):
        score = sum(features.values()) * 1.2
        return score, self._nivel(score)

    def random_forest(self, features):
        score = sum(features.values()) * 0.9
        return score, self._nivel(score)

    def svm(self, features):
        score = sum(features.values()) * 1.1
        return score, self._nivel(score)

    def _nivel(self, score):
        if score >= 15:
            return "ALTO"
        elif score >= 7:
            return "MEDIO"
        return "BAJO"