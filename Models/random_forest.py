from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns

class RandomForest:
    """La clase crea el modelo Random Forest para la clasificación del riesgo de suicidio
    de acuerdo a
    0 = Bajo
    1 = Mediano
    2 = Alto """
    
    def __init__(self):
        """Inicializa el modelo"""
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1,2),
                stop_words=None
            )),
            ("smote", SMOTE(random_state=42)),
            
            ("rf", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    def enntrenamientoModelo(self,df,text, label):
        y =df[label].astype(int).values
        """Segmenta los datos de entrenamiento y prueba, para entrenar el modelo"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df[text], y,
            test_size=0.3,
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
    
    def guardar_modelo(self, name):
        """ Guarda el modelo entrenado"""
        joblib.dump(self.model, f"{name}.pkl")
