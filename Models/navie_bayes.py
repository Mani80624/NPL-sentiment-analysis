from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class NavieBayes:
    """La clase crea el modelo Navie Bayes para la clasificación del riesgo de suicidio de acuerdo a
    0 -> Bajo
    1-> Mediano
    2-> Alto"""

    def __init__(self):
        """Inicializa el modelo"""
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            stop_words=None
            )),
            ("nb", MultinomialNB())
            ])

    def trainModel(self, df):
        """Segmenta los datos de entrenaminto y prueba y posteriormente entrena al modelo"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df["text"], df["label"], 
            test_size=0.2, 
            random_state=42)

        self.model.fit(self.X_train, self.y_train)

        return self.model
    
    def evaluate(self):
        """Imprime el rendimiento del modelo y la matriz de confusión"""
        y_pred = self.model.predict(self.X_test)
        print("Reporte de clasificación:\n")
        print(classification_report(self.y_test, y_pred))
        print("Matriz de confusión:\n")
        print(confusion_matrix(self.y_test, y_pred))

    def new_prediction(self, text):
        """Recibe una lista de textos y después entrega una predicción y una probalidad"""
        pred = self.model.predict(text)
        proba = self.model.predict_proba(text)

        return pred, proba
    
    def save_model(self, name):
        """Guarda el modelo entrenado para utilizarlo en el futuro"""
        joblib.dump(self.model, f"{name}.pkl")