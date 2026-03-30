import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from tools.load_data import load_dataset
from tools.vocab import build_vocab
from train_models.train_cnn import train_model
from tools.dataset import TextDataset
from Models.cnn_model import TextCNN


# =========================
# Evaluación por fold
# =========================
def evaluate_fold(model, X_test, y_test, vocab):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        TextDataset(X_test, y_test, vocab),
        batch_size=64
    )

    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro")
    }


# =========================
# Gráfica
# =========================
def plot_kfold(metrics):

    folds = list(range(1, len(metrics["accuracy"]) + 1))

    plt.style.use("seaborn-v0_8")

    plt.figure(figsize=(8, 5))

    plt.plot(folds, metrics["accuracy"], marker='o', label="Accuracy")
    plt.plot(folds, metrics["precision"], marker='o', label="Precision")
    plt.plot(folds, metrics["recall"], marker='o', label="Recall")
    plt.plot(folds, metrics["f1"], marker='o', label="F1-score")

    plt.grid(True)
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig("kfold_metrics.png")
    plt.show()


# =========================
# K-FOLD TRAINING
# =========================
def kfold_training(k=10):

    print("Starting K-Fold Cross Validation...")

    X, y = load_dataset("data/emotions_risk_scores_1.csv")

    # NO convertir X a numpy
    y = np.array(y)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    fold = 1

    for train_idx, test_idx in kf.split(X):

        print(f"\nFOLD {fold}")

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        # vocab SOLO con train
        vocab = build_vocab(X_train)

        # entrenar
        train_model(
            X_train, y_train,
            X_test, y_test,
            vocab,
            epochs=10,
            batch_size=64,
            patience=3
        )

        # cargar modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            "best_cnn_model.pth",
            map_location=device,
            weights_only=True
        )

        model = TextCNN(checkpoint["vocab_size"]).to(device)
        model.load_state_dict(checkpoint["model_state"])

        # evaluar
        results = evaluate_fold(model, X_test, y_test, vocab)

        for key in metrics:
            metrics[key].append(results[key])

        print(f"Fold {fold}: {results}")

        fold += 1

    print("\nFINAL RESULTS")
    print("Mean Accuracy:", np.mean(metrics["accuracy"]))
    print("Std:", np.std(metrics["accuracy"]))

    # gráfica final
    plot_kfold(metrics)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    kfold_training()