import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Librería esencial para el estilo visual de la imagen
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from tools.dataset import TextDataset
from Models.cnn_model import TextCNN

def plot_confusion(y_true, y_pred):
    """
    Genera una matriz de confusión con el estilo visual exacto:
    Fondo azul degradado, etiquetas 'Real'/'Predict' y valores centrados.
    """
    # 1. Calcular la matriz numérica
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Low", "Medium", "High"]

    # 2. Configurar el tamaño de la figura
    plt.figure(figsize=(8, 6))
    
    # 3. Crear el heatmap con Seaborn
    # annot=True: Muestra los números internos
    # fmt='d': Formato de número entero (evita 2.1e+03)
    # cmap='Blues': El esquema de color exacto de tu imagen
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=labels, 
                yticklabels=labels,
                annot_kws={"size": 12}) # Tamaño de fuente de los números

    # 4. Configurar etiquetas de ejes (idénticas a tu imagen)
    plt.ylabel('Real', fontsize=13, fontweight='bold')
    plt.xlabel('Predict', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix - TextCNN Evaluation', pad=20, fontsize=14)

    # 5. Ajuste y guardado
    plt.tight_layout()
    plt.savefig("confusion_matrix_styled.png", dpi=300)
    plt.show()

def evaluate(X_test, y_test):
    """
    Carga el modelo entrenado y realiza la evaluación sobre el conjunto de prueba.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en: {device}")

    # Cargar el checkpoint guardado durante el entrenamiento
    # Usamos el nombre del archivo generado por train_cnn.py
    try:
        checkpoint = torch.load(
            "best_cnn_model.pth", 
            map_location=device, 
            weights_only=False # Cambiar a False si el checkpoint contiene el dict del vocab
        )
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'best_cnn_model.pth'.")
        return

    vocab = checkpoint["vocab"]
    vocab_size = checkpoint["vocab_size"]

    # Reinstanciar la arquitectura TextCNN
    model = TextCNN(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Preparar el cargador de datos de prueba
    loader = DataLoader(
        TextDataset(X_test, y_test, vocab),
        batch_size=64,
        shuffle=False
    )

    y_true, y_pred = [], []

    # Inferencia sin cálculo de gradientes para ahorrar memoria
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            
            # Obtener el índice de la clase con mayor probabilidad
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # Mostrar métricas detalladas en consola
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))

    # Generar la gráfica estilizada
    plot_confusion(y_true, y_pred)

if __name__ == "__main__":
    # Nota: Este bloque es para pruebas locales si tienes los datos cargados
    print("Script de evaluación listo. Invocalo desde run_all_cnn.py")