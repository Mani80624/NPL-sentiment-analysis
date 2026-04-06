import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Models.cnn_model import TextCNN
from Models.CNN_scripts.dataset import TextDataset


# FOCAL LOSS (opcional)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss)


def train_model(X_train, y_train, X_val, y_val, vocab,
                epochs=25, batch_size=64, patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(
        TextDataset(X_train, y_train, vocab),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TextDataset(X_val, y_val, vocab),
        batch_size=batch_size
    )

    model = TextCNN(len(vocab)).to(device)

    # PESOS DE CLASE 
    class_weights = torch.tensor([1.0, 1.2, 1.5]).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = FocalLoss()  

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss += loss_fn(outputs, y).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # ===== EARLY STOPPING =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "vocab_size": len(vocab)
            }, "best_cnn_model.pth")

            print("Mejor modelo guardado")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping activado")
            break

    # ===== GRAFICA =====
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Curva de entrenamiento")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()