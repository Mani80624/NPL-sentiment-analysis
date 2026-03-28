import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.cnn_model import TextCNN
from tools.dataset import TextDataset
from tools.vocab import build_vocab


def train_model(tokenized_texts, labels, epochs=10, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    vocab = build_vocab(tokenized_texts)

    dataset = TextDataset(tokenized_texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextCNN(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for x, y in loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Guardado del modelo completo
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "vocab_size": len(vocab)
    }, "cnn_complete.pth")

    print("Modelo guardado como cnn_complete.pth")