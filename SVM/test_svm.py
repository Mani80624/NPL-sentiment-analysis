import re
import joblib


class TextPreprocessor:
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
    predictor = SVMEmotionPredictor("SVM/modelo_svm_emociones.pkl")

    test_texts = [
        "I feel happy and excited today",
        "I am lonely and broken inside",
        "I am afraid of the future",
        "I hate this situation so much",
        "I trust my family and feel safe",
        "I am surprised by the sudden news",
        "I am waiting for something important tomorrow"
    ]

    print("===== PRUEBAS DEL MODELO SVM =====")

    for text in test_texts:
        result = predictor.predict(text)

        print("\nTexto:", result["original_text"])
        print("Texto limpio:", result["clean_text"])
        print("Predicción:", result["prediction"])
        print("Probabilidades:")
        for emotion, prob in result["probabilities"].items():
            print(f"  {emotion}: {prob}%")