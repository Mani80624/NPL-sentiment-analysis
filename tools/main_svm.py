import joblib
from tools.data_loader_svm import DataLoader
from Models.text_preprocessor import TextPreprocessor
from Models.svm_model import SVMModel
from Models.trainer import Trainer
from Models.evaluator import Evaluator

from NLP.stopwords import StopWordsRemover


# =========================
# LOAD DATA
# =========================
loader = DataLoader()
df = loader.load_data("data/emotions_risk_scores_2.csv")

# =========================
# STOPWORDS
# =========================
sw = StopWordsRemover()
pre = TextPreprocessor(sw.get_stopwords())

# =========================
# MODEL
# =========================
model = SVMModel(pre.get_stopwords())

# =========================
# TRAIN
# =========================
trainer = Trainer(model)
X_test, y_test = trainer.train(df, "original_text", "riesgo")

# =========================
# EVALUATE
# =========================
eval = Evaluator()
eval.evaluate(model, X_test, y_test)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model.get_model(), "svm_text_optimized.pkl")

print("\n💾 Modelo guardado como svm_text_optimized.pkl")