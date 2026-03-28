from Models.cnn_model import TextCNN
from predictions.cnn_predictor import CNNPredictor

import torch


class ModelManager:
    """
    Orquestador de modelos de riesgo.

    Modelos disponibles:
    1 -> Naive Bayes
    2 -> CNN (Deep Learning)
    3 -> Random Forest
    4 -> SVM
    """

    def __init__(self):

        # Registro de modelos
        self.models = {
            1: self.naive_bayes,
            2: self.cnn,
            3: self.random_forest,
            4: self.svm
        }

        # =========================
        # ONFIGURACIÓN CNN
        # =========================

        self.vocab_size = 5000
        self.embed_dim = 100
        self.num_classes = 3  # BAJO, MEDIO, ALTO

        self.vocab = {}  # cargar vocab real aquí

        self.cnn_model = TextCNN(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes
        )

        # Cargar pesos entrenados cuando los tengas
        # self.cnn_model.load_state_dict(torch.load("models/cnn_model.pth"))

        self.cnn_model.eval()

        self.cnn_predictor = CNNPredictor(
            self.cnn_model,
            self.vocab
        )

    # =========================
    # MÉTODO PRINCIPAL
    # =========================

    def predict(self, model_id, features=None, tokens=None):
        """
        Ejecuta el modelo seleccionado.

        Parámetros:
        - model_id: int
        - features: dict (para modelos clásicos)
        - tokens: list (para CNN)

        Returns:
        - score, level
        """

        if model_id not in self.models:
            raise ValueError("Modelo no válido")

        # CNN usa tokens
        if model_id == 2:
            if tokens is None:
                raise ValueError("CNN necesita tokens")
            return self.cnn(tokens)

        # Modelos clásicos usan features
        if features is None:
            raise ValueError("Modelo clásico necesita features")

        return self.models[model_id](features)

    # =========================
    # MODELOS CLÁSICOS
    # =========================

    def naive_bayes(self, features):
        score = sum(features.values())
        return score, self._nivel(score)

    def random_forest(self, features):
        score = sum(features.values()) * 0.9
        return score, self._nivel(score)

    def svm(self, features):
        score = sum(features.values()) * 1.1
        return score, self._nivel(score)

    # =========================
    # CNN
    # =========================

    def cnn(self, tokens):
        return self.cnn_predictor.predict(tokens)

    # =========================
    # FUNCIÓN DE NIVEL
    # =========================

    def _nivel(self, score):
        """
        Convierte score a nivel de riesgo
        """

        if score >= 15:
            return "ALTO"
        elif score >= 7:
            return "MEDIO"
        return "BAJO"