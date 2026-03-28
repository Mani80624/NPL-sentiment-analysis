import torch
from Models.cnn_model import TextCNN


class CNNPredictor:

    def __init__(self, model_path, max_len=50):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.vocab = checkpoint["vocab"]

        self.model = TextCNN(vocab_size=checkpoint["vocab_size"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        self.model.eval()

        self.max_len = max_len

    def encode(self, tokens):

        indices = [self.vocab.get(t, 0) for t in tokens]

        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def predict(self, tokens):

        x = self.encode(tokens).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            pred = torch.argmax(logits, dim=1).item()

        niveles = ["BAJO", "MEDIO", "ALTO"]

        return pred, niveles[pred]