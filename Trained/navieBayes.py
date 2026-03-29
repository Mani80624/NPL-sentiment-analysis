from Models.navie_bayes import NavieBayes
from NLP.normalizacion import TextNormalizer
from NLP.tokenizer import Tokenizer
from NLP.stopwords import StopWordsRemover
from pathlib import Path
import pandas as pd
import numpy as np

navie_bayes = NavieBayes()
normalizacion = TextNormalizer()
tokenizar = Tokenizer()
stopwords_remove = StopWordsRemover()

path = Path("C:/Users/ma-nu/Downloads/sentiment-analysis/NPL-sentiment-analysis/emotions_risk_scores_1.csv")
path_modified = Path("C:/Users/ma-nu/Downloads/minería de datos/emotions_risk_scores_1.csv")
df = pd.read_csv(path_modified)
trained_select = df[['original_text', 'riesgo']]
trained_select['riesgo'] = trained_select['riesgo'].replace({"Bajo":0, "Medio": 1, "Alto":2})


trained_select['original_text'] = trained_select['original_text'].apply(normalizacion.clean_text)

#navie_bayes.trainModel(trained_select, "original_text", "riesgo")
#navie_bayes.evaluate()
#navie_bayes.plot_confusion_matrix()
#navie_bayes.save_model("Trained/NB_without_stopWords_2")
navie_bayes.cross_validation(trained_select, "original_text", "riesgo",10)

navie_bayes.plot_cv_results()