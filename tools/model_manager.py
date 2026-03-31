import joblib
import torch
import torch.nn.functional as F
#import numpy as np

class CNNsklearnWrapper:
    def __init__(self, model, vocab, max_len=150):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len

    def _preprocess(self, text_raw):
        """Convierte el texto en tensores de índices para la CNN"""
        if not isinstance(text_raw, str): 
            text_raw = ""
        tokens = text_raw.lower().split()
        indices = [self.vocab.get(t, 0) for t in tokens]
        
        # Aplicar padding o truncado según el entrenamiento
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor([indices], dtype=torch.long)

    def predict(self, text_raw):
        """Este es el método que busca el ModelManager"""
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess(text_raw)
            outputs = self.model(x)
            return torch.argmax(outputs, dim=1).numpy()

    def predict_proba(self, text_raw):
        """Necesario para calcular la confianza/score"""
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess(text_raw)
            outputs = self.model(x)
            return F.softmax(outputs, dim=1).numpy()

class ModelManager:

    def __init__(self):
        self.models_paths = {
            1: "Trained/NB_without_stopWords_2.pkl",
            2: "Trained/cnn_full_model.pkl",
            3: "Trained/RF_without_stopWords_2.pkl",
            4: "Trained/SVM_without_stopWords_2.pkl"
        }
        
        # Cargar los modelos
        self.models = {}
        for model_id, path in self.models_paths.items():
            try:
                self.models[model_id] = joblib.load(path)
                print(f"Modelo {model_id} cargado correctamente desde {path}")
            except FileNotFoundError:
                print(f"Advertencia: Modelo {path} no encontrado")
                self.models[model_id] = None

    def predict(self, model_id, features, raw_text=""):

        if model_id not in self.models or self.models[model_id] is None:
            raise ValueError("Modelo no válido o no cargado")

        model = self.models[model_id]
        
        # Obtener predicción del modelo (0, 1, 2)
        try:
            if model_id == 2: #CNN
                prediction = model.predict(raw_text)[0]
                probabilities = model.predict_proba(raw_text)[0]
            else: # NB, RF, SVM
                input_data = [raw_text]
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
            
            confidence = max(probabilities)

        except Exception as e:
            print(f"Error en predicción: {e}")
            prediction = 0
            confidence = 0.0
        
        # Convertir predicción a nivel de riesgo
        risk_level = self._nivel(prediction)
        
        return float(confidence), risk_level

    def _nivel(self, prediction):
        """Mapea la predicción del modelo a nivel de riesgo"""
        if prediction == 0:
            return "BAJO"
        elif prediction == 1:
            return "MEDIANO"
        elif prediction == 2:
            return "ALTO"
        return "BAJO"