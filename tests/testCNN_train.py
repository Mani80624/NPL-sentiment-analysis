from train_models.train_cnn import train_model

# Dataset de prueba 
texts = [
    ["i", "feel", "sad", "and", "alone"],
    ["life", "is", "beautiful"],
    ["i", "want", "to", "die"],
    ["i", "am", "happy"],
    ["everything", "is", "hopeless"],
    ["i", "love", "my", "life"]
]

# 0 = BAJO, 1 = MEDIO, 2 = ALTO
labels = [2, 0, 2, 0, 2, 0]

train_model(texts, labels, epochs=5)