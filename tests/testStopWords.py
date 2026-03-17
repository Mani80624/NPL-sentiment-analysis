from NLP.stopwords import StopWordsRemover

stop_word = StopWordsRemover()

tokens = [
    'i','feel','terrible','and','i','want','to','end',
    'my','life','one','way','or','another'
    ]
results = stop_word.remove(tokens)
print(results)