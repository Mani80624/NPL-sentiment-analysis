import torch
from Models.cnn_model import TextCNN


class CNNPredictor:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.vocab = checkpoint["vocab"]

        self.model = TextCNN(checkpoint["vocab_size"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def encode(self, tokens):

        indices = [self.vocab.get(t, 0) for t in tokens][:50]
        indices += [0] * (50 - len(indices))

        return torch.tensor(indices).unsqueeze(0)

    def predict(self, tokens):

        x = self.encode(tokens).to(self.device)

        with torch.no_grad():
            pred = torch.argmax(self.model(x), dim=1).item()

        return pred, ["BAJO", "MEDIO", "ALTO"][pred]