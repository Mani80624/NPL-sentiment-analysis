import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


class SVMEmotionClassifier:
    """Clasificador de emociones basado en TF-IDF + SVM."""

    def __init__(self, max_features: int = 5000, ngram_range=(1, 2), kernel: str = "linear"):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )),
            ("svm", SVC(
                kernel=kernel,
                probability=True
            ))
        ])
        self.preprocessor = TextPreprocessor()

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        df["text"] = df["text"].apply(self.preprocessor.clean_text)
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        X = df["text"]
        y = df["label"]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.model.predict(X_test)

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        return results

    def save_model(self, model_path: str) -> None:
        joblib.dump(self.model, model_path)

    def predict(self, text: str) -> dict:
        clean_text = self.preprocessor.clean_text(text)
        prediction = self.model.predict([clean_text])[0]
        probabilities = self.model.predict_proba([clean_text])[0]
        classes = self.model.named_steps["svm"].classes_

        return {
            "text": text,
            "clean_text": clean_text,
            "prediction": prediction,
            "probabilities": {
                cls: round(prob * 100, 2) for cls, prob in zip(classes, probabilities)
            }
        }


if __name__ == "__main__":
    DATASET_PATH = "SVM/plutchik_dataset.csv"
    MODEL_PATH = "SVM/modelo_svm_emociones.pkl"

    classifier = SVMEmotionClassifier()

    df = classifier.load_dataset(DATASET_PATH)

    print("Primeras filas del dataset:")
    print(df.head())

    print("\nColumnas:")
    print(df.columns)

    print("\nCantidad por clase:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = classifier.split_data(df)

    print("\nCantidad entrenamiento:", len(X_train))
    print("Cantidad prueba:", len(X_test))

    classifier.train(X_train, y_train)
    print("\nModelo entrenado correctamente.")

    results = classifier.evaluate(X_test, y_test)

    print("\nAccuracy:")
    print(results["accuracy"])

    print("\nClassification Report:")
    print(results["classification_report"])

    print("\nMatriz de confusión:")
    print(results["confusion_matrix"])

    classifier.save_model(MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")