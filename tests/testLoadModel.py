import joblib
path = "C:/Users/ma-nu/Downloads/minería de datos/modelos entrenados/svm_model.pkl"


modelo = joblib.load(path)
print(modelo)