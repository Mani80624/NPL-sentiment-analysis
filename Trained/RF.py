from Models.random_forest import RandomForest
from  NLP.tokenizer import Tokenizer
from NLP.normalizacion import TextNormalizer
from NLP.stopwords import StopWordsRemover
from pathlib import Path
import pandas as pd

# Entrenamiento real de Random Forest con el dataset
random_forest = RandomForest()
normalizacion = TextNormalizer()
tokenizar = Tokenizer()
stopwords_remove =StopWordsRemover()

path = Path("C:/Users/Asant/Documents/Maestria/Proyecto_articulo/emotions_risk_scores_1.csv")
df = pd.read_csv(path)
trained_select = df[['original_text', 'riesgo']]
trained_select['riesgo'] = trained_select['riesgo'].replace({"Bajo":0, "Medio": 1, "Alto":2})

trained_select['original_text'] = trained_select['original_text'].apply(normalizacion.clean_text)

random_forest.entrenamientoModelo(trained_select, "original_text", "riesgo")
random_forest.evaluacion()

random_forest.plot_confusion_matrix()
random_forest.guardar_modelo("Trained/RF_without_stopWords_2")
random_forest.cross_validation(trained_select, "original_text", "riesgo",10)

random_forest.plot_cv_results()