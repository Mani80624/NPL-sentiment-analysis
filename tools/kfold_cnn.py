import numpy as np
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
# K-FOLD TRAINING
# =========================
def kfold_training(k=10):
    print(f"Iniciando {k}-Fold Cross Validation...")

    X, y = load_dataset("data/emotions_risk_scores_1.csv")
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
        print(f"\n--- TRABAJANDO EN FOLD {fold} ---")

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Vocabulario específico del fold
        vocab = build_vocab(X_train)

        # Entrenar (Carga el modelo interno según tu configuración)
        train_model(
            X_train, y_train,
            X_test, y_test,
            vocab,
            epochs=10,
            batch_size=64,
            patience=3
        )

        # Cargar el mejor modelo guardado en este fold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            "best_cnn_model.pth",
            map_location=device,
            weights_only=True
        )

        model = TextCNN(checkpoint["vocab_size"]).to(device)
        model.load_state_dict(checkpoint["model_state"])

        # Evaluar
        results = evaluate_fold(model, X_test, y_test, vocab)

        for key in metrics:
            metrics[key].append(results[key])

        print(f"Resultados Fold {fold}: Acc: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        fold += 1

    # =========================
    # REPORTE FINAL (SIN PLOT)
    # =========================
    print("\n" + "="*50)
    print("RESUMEN DETALLADO POR FOLD")
    print("="*50)
    print(f"{'Fold':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 50)
    
    for i in range(k):
        print(f"{i+1:<8} | {metrics['accuracy'][i]:.4f}   | {metrics['precision'][i]:.4f}   | {metrics['recall'][i]:.4f}   | {metrics['f1'][i]:.4f}")
    
    print("-" * 50)
    print(f"{'PROMEDIO':<8} | {np.mean(metrics['accuracy']):.4f}   | {np.mean(metrics['precision']):.4f}   | {np.mean(metrics['recall']):.4f}   | {np.mean(metrics['f1']):.4f}")
    print(f"{'DESV. ST':<8} | {np.std(metrics['accuracy']):.4f}   | {np.std(metrics['precision']):.4f}   | {np.std(metrics['recall']):.4f}   | {np.std(metrics['f1']):.4f}")
    print("="*50)


if __name__ == "__main__":
    kfold_training()