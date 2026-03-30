import joblib

class ModelManager:

    def __init__(self):
        self.models_paths = {
            1: "Trained/NB_without_stopWords_2.pkl",
            2: "Trained/CNN_without_stopWords_2.pkl",
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

    def predict(self, model_id, features):
        if model_id not in self.models or self.models[model_id] is None:
            raise ValueError("Modelo no válido o no cargado")

        model = self.models[model_id]
        
        # Obtener predicción del modelo (0, 1, 2)
        try:
            prediction = model.predict([list(features.values())])[0]
            probabilities = model.predict_proba([list(features.values())])[0]
            
            confidence = max(probabilities)

        except Exception as e:
            print(f"Error en predicción: {e}")
            prediction = 0
            confidence = 0.0
        
        # Convertir predicción a nivel de riesgo
        risk_level = self._nivel(prediction)
        
        return confidence, risk_level

    def _nivel(self, prediction):
        """Mapea la predicción del modelo a nivel de riesgo"""
        if prediction == 0:
            return "BAJO"
        elif prediction == 1:
            return "MEDIANO"
        elif prediction == 2:
            return "ALTO"
        return "BAJO"