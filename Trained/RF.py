from Models.random_forest import RandomForest
from  NLP.tokenizer import Tokenizer
from NLP.normalizacion import TextNormalizer
from NLP.stopwords import StopWordsRemover
from pathlib import Path
import pandas as pd


random_forest = RandomForest()
normalizacion = TextNormalizer()
tokenizar = Tokenizer()
stopwords_remove =StopWordsRemover()

path = Path("C:/Users/Asant/Documents/Maestria/Proyecto_articulo/emotions_risk_scores_1.csv")
df = pd.read_csv(path)
trained_select = df[['original_text', 'riesgo']]
trained_select['riesgo'] = trained_select['riesgo'].replace({"Bajo":0, "Medio": 1, "Alto":2})

trained_select['original_text'] = trained_select['original_text'].apply(normalizacion.clean_text)

random_forest.enntrenamientoModelo(trained_select, "original_text", "riesgo")
random_forest.evaluacion()