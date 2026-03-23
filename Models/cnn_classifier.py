import torch
from models.cnn_emotion_model import EmotionCNN

class CNNClassifier:
    def __init__(self, model_path="models/emotion_cnn.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EmotionCNN(num_features=7, num_classes=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, emotion_vector):
        tensor = torch.tensor(emotion_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).item()

        return pred