from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from NLP.stopwords   import  StopWordsRemover
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import numpy as np

class RandomForest:
    """La clase crea el modelo Random Forest para la clasificación del riesgo de suicidio
    de acuerdo a
    0 = Bajo
    1 = Mediano
    2 = Alto """
    
    def __init__(self):
        """Inicializa el modelo"""
        self.stop_words = StopWordsRemover()
        self.stop_words_update = self.stop_words.get_stopwords()
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1,2),
                stop_words=list(self.stop_words_update),
                max_features=5000
            )),
            ("select", SelectKBest(chi2, k=3000)),
            ("smote", SMOTE(random_state=42)),
            
            ("rf", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    def entrenamientoModelo(self,df,text, label):
        y =df[label].astype(int).values
        """Segmenta los datos de entrenamiento y prueba, para entrenar el modelo"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df[text], y,
            test_size=0.2,
            random_state=42,
            stratify=y)
        
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def evaluacion(self):
        """Imprime el rendimiento del modelo y la matríz de confusión"""
        y_pred = self.model.predict(self.X_test)
        print("Reporte de clasificación: \n")
        print(classification_report(self.y_test,y_pred))
        print("Matríz de confusión: \n")
        print(confusion_matrix(self.y_test,y_pred))
        
    def prediccion(self, text):
        """Entrega una nueva prediccón y probabilidad"""
        pred = self.model.predict(text)
        proba = self.model.predict_proba(text)
        return pred, proba
    
    def cross_validation(self, df, texts, labels, folds=5):
        """Realiza validación cruzada estratificada"""
        y = df[labels].astype(int).values
        X = df[texts]

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        # Evaluar varias métricas a la vez
        self.scores = cross_validate(
            self.model, X, y,
            cv=cv,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        )

        print("Resultados por fold:")
        for metric, values in self.scores.items():
            if "test" in metric:
                print(f"{metric}: {values}")

        print("\nPromedios:")
        for metric, values in self.scores.items():
            if "test" in metric:
                print(f"{metric}: {np.mean(values):.3f}")
        return self.scores
    
    
    def plot_cv_results(self):
        """Grafica los resultados de cada fold"""
        if not hasattr(self, "scores"):
            raise ValueError("Primero ejecuta cross_validation()")
        
        metrics = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]  # Nuevas etiquetas

        plt.figure(figsize=(10,6))
        for metric, label in zip(metrics, labels):
            plt.plot(self.scores[metric], marker="o", label=label)

        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_confusion_matrix(self):
        """Genera un gráfico de la matriz de confusión"""
        if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
            raise ValueError("Primero entrena y evalúa el modelo con trainModel() y evaluate()")

        # Predicciones
        y_pred = self.model.predict(self.X_test)

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)

        # Etiquetas de clases
        labels = ["Low", "Medium", "High"]

        # Gráfico con seaborn
        plt.figure(figsize=(7,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)

        #plt.title("Matriz de Confusión")
        plt.xlabel("Predict")
        plt.ylabel("Real")
        plt.show()
    
    def guardar_modelo(self, name):
        """ Guarda el modelo entrenado"""
        joblib.dump(self.model, f"{name}.pkl")
