from NLP.plotPlutchik import Plot

pluchik_1 = Plot()

emociones = {
    "joy": 0.7,
    "trust": 0.5,
    "fear": 0.2,
    "surprise": 0.3,
    "sadness": 0.1,
    "disgust": 0.05,
    "anger": 0.4,
    "anticipation": 0.6
}


pluchik_1.plot(emociones)