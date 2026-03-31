from NLP.risk_pipeline import RiskDetectionPipeline
from predictions.cnn_predictor import CNNPredictor

pipeline = RiskDetectionPipeline()
cnn = CNNPredictor("cnn_complete.pth")

text = "I feel empty and alone, I want to disappear"

# 🔥 pipeline NLP
result = pipeline.process(text)

tokens = result["tokens"]

# 🔥 CNN
pred, level = cnn.predict(tokens)

print("=== RESULTADO COMPLETO ===")
print("Texto:", text)
print("Tokens:", tokens)
print("Emoción dominante:", result["dominant_emotion"])
print("CNN Riesgo:", level)