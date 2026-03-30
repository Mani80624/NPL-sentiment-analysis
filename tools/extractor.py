import torch
import pickle
from Models.cnn_model import TextCNN

def export_to_pickle(checkpoint_path, output_pkl_path):
    # 1. Definir dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Cargar el checkpoint (contiene vocab_size y state_dict)
    # Usamos weights_only=False porque el checkpoint original tiene el dict del vocabulario
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab_size = checkpoint["vocab_size"]
    model_state = checkpoint["model_state"]
    vocab = checkpoint["vocab"]

    # 3. Crear instancia vacía de la arquitectura
    model = TextCNN(vocab_size=vocab_size)
    
    # 4. Cargar el state_dict en la instancia
    model.load_state_dict(model_state)
    model.to(device)
    model.eval() # Configurar para inferencia

    # 5. Empaquetar modelo y metadatos necesarios (como el vocabulario)
    data_to_save = {
        "model": model,
        "vocab": vocab,
        "architecture_config": {
            "vocab_size": vocab_size,
            "embed_dim": 200,
            "num_classes": 3
        }
    }

    # 6. Guardar como archivo .pkl
    with open(output_pkl_path, "wb") as f:
        pickle.dump(data_to_save, f)
    
    print(f"Modelo exportado exitosamente a: {output_pkl_path}")

if __name__ == "__main__":
    # Asegúrate de que el nombre del archivo coincida con el que generó train_cnn.py
    export_to_pickle("cnn_model_M_K.pth", "cnn_full_model.pkl")