from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class SVM:
    """La clase crea el modelo SVM para la clasificación de emociones
    de acuerdo al modelo de Plutchik:
    joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    """

    def __init__(self):
        """Inicializa el modelo SVM"""
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=None
            )),
            ("svm", SVC(
                kernel="linear",
                probability=True,
                random_state=42
            ))
        ])

    def enntrenamienoModelo(self, df):
        """Segmenta los datos de entrenamiento y prueba, para entrenar el modelo"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.3,
            random_state=42
        )

        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluacion(self):
        """Imprime el rendimiento del modelo y la matriz de confusión"""
        y_pred = self.model.predict(self.X_test)
        print("Reporte de clasificación:\n")
        print(classification_report(self.y_test, y_pred))
        print("Matriz de confusión:\n")
        print(confusion_matrix(self.y_test, y_pred))

    def prediccion(self, text):
        """Entrega una nueva predicción y probabilidad"""
        pred = self.model.predict(text)
        proba = self.model.predict_proba(text)
        return pred, proba

    def guardar_modelo(self, name):
        """Guarda el modelo entrenado"""
        joblib.dump(self.model, f"{name}.pkl")