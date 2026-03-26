import re
import joblib


class TextPreprocessor:
    """Clase encargada del preprocesamiento del texto."""

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\sáéíóúñ']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class SVMEmotionPredictor:
    """Clase encargada de cargar el modelo y hacer predicciones."""

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = TextPreprocessor()

    def predict(self, text: str) -> dict:
        clean_text = self.preprocessor.clean_text(text)
        prediction = self.model.predict([clean_text])[0]
        probabilities = self.model.predict_proba([clean_text])[0]
        classes = self.model.named_steps["svm"].classes_

        return {
            "original_text": text,
            "clean_text": clean_text,
            "prediction": prediction,
            "probabilities": {
                cls: round(prob * 100, 2) for cls, prob in zip(classes, probabilities)
            }
        }


if __name__ == "__main__":
    MODEL_PATH = "SVM/modelo_svm_emociones.pkl"

    predictor = SVMEmotionPredictor(MODEL_PATH)

    text = input("Escribe un texto en inglés: ")
    result = predictor.predict(text)

    print("\nTexto limpio:", result["clean_text"])
    print("Emoción predicha:", result["prediction"])

    print("\nProbabilidades:")
    for emotion, prob in result["probabilities"].items():
        print(f"{emotion}: {prob}%")