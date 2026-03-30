import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SVM:
    def __init__(self):
        self.features = [
            "anger_count",
            "anticipation_count",
            "disgust_count",
            "fear_count",
            "joy_count",
            "sadness_count",
            "surprise_count",
            "trust_count",
            "text_length",
            "score",
            "score_norm",
            "score_wins"
        ]

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ])

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def entrenamientoModelo(self, df):
        if "riesgo" not in df.columns:
            raise ValueError("No existe la columna 'riesgo' en el dataset.")

        X = df.loc[:, self.features].copy()
        y = df["riesgo"].copy()

        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,   # 80% entrenamiento, 20% prueba
            random_state=42,
            stratify=y
        )

        self.model.fit(self.X_train, self.y_train)

    def evaluacion(self, output_dir="results"):
        if self.X_test is None:
            raise ValueError("Primero debes entrenar el modelo.")

        os.makedirs(output_dir, exist_ok=True)

        self.y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)

        # Orden real del dataset
        clases_es = ["Bajo", "Medio", "Alto"]

        # Etiquetas en inglés para mostrar
        clases_en = ["Low", "Medium", "High"]

        cm = confusion_matrix(self.y_test, self.y_pred, labels=clases_es)

        print("\nAccuracy:")
        print(acc)

        print("\nReporte de clasificación:")
        print(report)

        print("\nMatriz de confusión:")
        print(cm)

        # Guardar reporte
        with open(os.path.join(output_dir, "svm_report.txt"), "w", encoding="utf-8") as f:
            f.write("Accuracy:\n")
            f.write(str(acc))
            f.write("\n\nClassification Report:\n")
            f.write(report)

        # Guardar CSV de matriz
        cm_df = pd.DataFrame(cm, index=clases_en, columns=clases_en)
        cm_df.to_csv(os.path.join(output_dir, "svm_confusion_matrix.csv"), encoding="utf-8")

        # Guardar heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=clases_en,
            yticklabels=clases_en
        )
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "svm_confusion_matrix.png"), dpi=300)
        plt.close()

        return {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }

    def validacion_cruzada(self, df, folds=10):
        if "riesgo" not in df.columns:
            raise ValueError("No existe la columna 'riesgo' en el dataset.")

        X = df.loc[:, self.features].copy()
        y = df["riesgo"].copy()

        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(0)

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        scores = cross_val_score(self.model, X, y, cv=kf, n_jobs=1)

        print("\nK-Fold Cross Validation:")
        print(scores)

        print("\nMean accuracy:")
        print(scores.mean())

        print("\nStandard deviation:")
        print(scores.std())

        return scores

    def grafica_validacion_cruzada(self, df, folds=10, output_dir="results"):
        if "riesgo" not in df.columns:
            raise ValueError("No existe la columna 'riesgo' en el dataset.")

        os.makedirs(output_dir, exist_ok=True)

        X = df.loc[:, self.features].copy()
        y = df["riesgo"].copy()

        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(0)

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        scoring = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }

        scores = cross_validate(
            self.model,
            X,
            y,
            cv=kf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False
        )

        folds_x = list(range(1, folds + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(folds_x, scores["test_accuracy"], marker="o", label="test_accuracy")
        plt.plot(folds_x, scores["test_precision_macro"], marker="o", label="test_precision_macro")
        plt.plot(folds_x, scores["test_recall_macro"], marker="o", label="test_recall_macro")
        plt.plot(folds_x, scores["test_f1_macro"], marker="o", label="test_f1_macro")

        plt.title("Cross-Validation Results SVM per Fold")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.xticks(folds_x)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "svm_cross_validation_per_fold.png"), dpi=300)
        plt.close()

        df_scores = pd.DataFrame({
            "fold": folds_x,
            "test_accuracy": scores["test_accuracy"],
            "test_precision_macro": scores["test_precision_macro"],
            "test_recall_macro": scores["test_recall_macro"],
            "test_f1_macro": scores["test_f1_macro"]
        })

        df_scores.to_csv(
            os.path.join(output_dir, "svm_cross_validation_per_fold.csv"),
            index=False,
            encoding="utf-8"
        )

        print("\nCross-validation per fold:")
        print(df_scores)

        print("\nAverages:")
        print(df_scores.drop(columns=["fold"]).mean())

        return df_scores

    def prediccion(self, data):
        data = data.loc[:, self.features].copy()

        for col in self.features:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.fillna(0)

        pred = self.model.predict(data)
        prob = self.model.predict_proba(data)

        return pred, prob

    def guardar_modelo(self, nombre):
        if not nombre.endswith(".pkl"):
            nombre += ".pkl"

        joblib.dump(self.model, nombre)