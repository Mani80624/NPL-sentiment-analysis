import torch
import pickle

# cargar archivo .pth
checkpoint = torch.load("cnn_model_M_K.pth", map_location="cpu")

# guardar como .pkl
with open("cnn_model.pkl", "wb") as f:
    pickle.dump(checkpoint, f)

print("Convertido a .pkl")