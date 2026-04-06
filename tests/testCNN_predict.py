from Models.CNN_scripts.cnn_predictor import CNNPredictor

predictor = CNNPredictor("cnn_complete.pth")

tests = [
    ["i", "feel", "empty"],
    ["i", "am", "very", "happy"],
    ["i", "want", "to", "disappear"]
]

for t in tests:
    pred, level = predictor.predict(t)
    print(f"Texto: {t}")
    print(f"Predicción: {level} ({pred})\n")