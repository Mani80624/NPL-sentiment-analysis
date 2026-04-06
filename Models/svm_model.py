import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optimización Intel
from sklearnex import patch_sklearn
patch_sklearn()

from NLP.stopwords import StopWordsRemover

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Memory


class SVMText:
    def __init__(self):
        stopwords = StopWordsRemover()
        self.stopwords_e = stopwords.get_stopwords()

        self.text_column = "original_text"
        self.label_column = "riesgo"

        memory = Memory(location="cache_dir", verbose=0)

        self.model = ImbPipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2,
                max_df=0.95,
                dtype=np.float32,
                stop_words=list(self.stopwords_e)
            )),
            ("svm", SVC(
                kernel="rbf",
                C=1.0,
                gamma = "scale",
                class_weight="balanced",
                cache_size=4000 #Uso de la RAM
            ))
        ], memory=memory)

        self.cv_scores = None
        self.cv_scores_df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    # Preparación de datos
    def preparar_datos(self, df):
        if self.text_column not in df.columns:
            raise ValueError(f"No existe la columna '{self.text_column}'")

        if self.label_column not in df.columns:
            raise ValueError(f"No existe la columna '{self.label_column}'")

        data = df[[self.text_column, self.label_column]].copy()
        data[self.text_column] = data[self.text_column].fillna("").astype(str)
        data[self.label_column] = data[self.label_column].fillna("").astype(str)

        data = data[data[self.text_column].str.strip() != ""]
        data = data[data[self.label_column].str.strip() != ""]

        return data

    # Validación cruzada
    def validacion_cruzada(self, df, folds=5):
        data = self.preparar_datos(df)

        X = data[self.text_column]
        y = data[self.label_column]

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        scoring = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }

        self.cv_scores = cross_validate(
            self.model,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        self.cv_scores_df = pd.DataFrame({
            "fold": list(range(1, folds + 1)),
            "accuracy": self.cv_scores["test_accuracy"],
            "precision": self.cv_scores["test_precision_macro"],
            "recall": self.cv_scores["test_recall_macro"],
            "f1": self.cv_scores["test_f1_macro"]
        })

        print(self.cv_scores_df)
        print(self.cv_scores_df.mean(numeric_only=True))

        return self.cv_scores_df

    # Gráfica de validación cruzada
    def grafica_validacion_cruzada(self, output_dir="results_text"):
        if self.cv_scores_df is None:
            raise ValueError("Ejecuta validacion_cruzada() primero")

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.cv_scores_df["fold"], self.cv_scores_df["accuracy"], marker="o", label="Accuracy")
        plt.plot(self.cv_scores_df["fold"], self.cv_scores_df["precision"], marker="o", label="Precision")
        plt.plot(self.cv_scores_df["fold"], self.cv_scores_df["recall"], marker="o", label="Recall")
        plt.plot(self.cv_scores_df["fold"], self.cv_scores_df["f1"], marker="o", label="F1")

        plt.title("Resultados CV por Fold")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cv_results.png", dpi=300)
        plt.show()

        self.cv_scores_df.to_csv(f"{output_dir}/cv_results.csv", index=False)

    # Entrenamiento
    def entrenamientoModelo(self, df):
        data = self.preparar_datos(df)

        X = data[self.text_column]
        y = data[self.label_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model.fit(self.X_train, self.y_train)

    # Evaluación
    def evaluacion(self):
        if self.X_test is None:
            raise ValueError("Ejecuta entrenamientoModelo() primero")

        self.y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)

        print("Accuracy:", acc)
        print(report)

        labels = sorted(self.y_test.unique())
        cm = confusion_matrix(self.y_test, self.y_pred, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)

        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")

        os.makedirs("results_text", exist_ok=True)
        plt.savefig("results_text/confusion_matrix.png", dpi=300)
        plt.show()

        return acc, report, cm

    # Predicción
    def prediccion(self, textos):
        textos = pd.Series(textos).fillna("").astype(str)
        return self.model.predict(textos)

    # Guardar modelo
    def guardar_modelo(self, nombre="svm_text_model.pkl"):
        joblib.dump(self.model, nombre)
        print(f"Modelo guardado en {nombre}")