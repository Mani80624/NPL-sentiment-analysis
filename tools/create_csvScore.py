from NLP.risk_features import RiskFeatureExtractor
from NLP.normalizacion import TextNormalizer
from NLP.tokenizer import Tokenizer
from NLP.stopwords import StopWordsRemover
from pathlib import Path
import pandas as pd


class GetCSVCount:
    def __init__(self, path_file, column_text):
        self.risk_features = RiskFeatureExtractor("C:/Users/ma-nu/Downloads/sentiment-analysis/NPL-sentiment-analysis/Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
        self.normalization = TextNormalizer()
        self.tokenizer = Tokenizer()
        self.stopWords = StopWordsRemover()

        self.path_file = Path(path_file)
        self.column_text = column_text

        if self.path_file.suffix == '.xls':
            self.read_df = pd.read_excel(self.path_file, engine="xlrd")
        elif self.path_file.suffix == '.csv':
            self.read_df=pd.read_csv(self.path_file)

    def _preprocesing(self):
        texts = self.read_df[self.column_text]
        texts = texts.apply(self.normalization.clean_text)
        texts = texts.apply(self.tokenizer.tokenize)
        texts = texts.apply(self.stopWords.remove)

        return texts

    def counter_emotion(self):
        counts_emo = self._preprocesing().apply(self.risk_features.extract)
        text_cat = self.read_df[self.column_text]
        text_cat_df = pd.json_normalize(counts_emo)
        text_cat_df["original_text"] = text_cat.reset_index(drop=True)
        #text_cat_df.pop('text_length')
        return text_cat_df


    def save_csv(self,name):
        self.counter_emotion().to_csv(f"C:/Users/ma-nu/Downloads/minería de datos/{name}.csv", 
                                      index=False, 
                                      encoding='utf-8')
        print(f"Se creo el archivo {name} con los conteos de cada emocioón")


path = "C:/Users/ma-nu/Downloads/Combined Data/Combined Data.csv"
text_column = "statement"
getcsv = GetCSVCount(path, text_column)

getcsv.save_csv("counts_leghts_chavos")