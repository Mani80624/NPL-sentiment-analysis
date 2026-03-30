import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from tools.dataset import TextDataset
from Models.cnn_model import TextCNN


def plot_confusion(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    labels = ["Low", "Medium", "High"]

    plt.figure(figsize=(6, 6))
    plt.imshow(cm)

    # ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # valores dentro de la matriz
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def evaluate(X_test, y_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cargar modelo seguro
    checkpoint = torch.load(
        "cnn_model_M_K.pth",
        map_location=device,
        weights_only=True
    )

    vocab = checkpoint["vocab"]

    model = TextCNN(checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        TextDataset(X_test, y_test, vocab),
        batch_size=64
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    # plot bonito
    plot_confusion(y_true, y_pred)