from tools.load_data import load_dataset
from tools.data_split import split_data
from tools.vocab import build_vocab
from train_models.train_cnn import train_model
from tools.evaluate_cnn import evaluate


X, y = load_dataset("data/emotions_risk_scores_1.csv")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print(f"Train: {len(X_train)}")
print(f"Val: {len(X_val)}")
print(f"Test: {len(X_test)}")

vocab = build_vocab(X_train)

train_model(
    X_train, y_train,
    X_val, y_val,
    vocab,
    epochs=50,
    batch_size=64,
    patience=8
)

evaluate(X_test, y_test)