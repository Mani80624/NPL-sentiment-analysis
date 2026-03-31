import torch
import joblib
from tools.model_manager import CNNsklearnWrapper
from Models.cnn_model import TextCNN

def save_correct_pickle():
    # 1. Cargar el checkpoint original de PyTorch
    checkpoint = torch.load("cnn_model_M_K.pth", map_location="cpu")
    
    # 2. Reconstruir la arquitectura
    model = TextCNN(vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state"])
    
    # 3. Crear el Wrapper con los métodos predict/predict_proba actualizados
    wrapped_model = CNNsklearnWrapper(
        model=model, 
        vocab=checkpoint["vocab"],
        max_len=150
    )
    
    # 4. Guardar en la carpeta Trained
    joblib.dump(wrapped_model, "Trained/cnn_full_model.pkl")
    print("Archivo .pkl regenerado con el atributo 'predict' correctamente.")

if __name__ == "__main__":
    save_correct_pickle()