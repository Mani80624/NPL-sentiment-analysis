from tools.load_data import load_dataset
from tools.data_split import split_data
from tools.evaluate_cnn import evaluate


# cargar datos
X, y = load_dataset("data/emotions_risk_scores_2.csv")

# hacer split (igual que entrenamiento)
_, _, X_test, _, _, y_test = split_data(X, y)

# evaluar
evaluate(X_test, y_test)