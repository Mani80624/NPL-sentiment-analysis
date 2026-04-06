import pandas as pd
from Models.svm_model import SVMText


def main():
    print("Cargando dataset...")

    # Ruta del dataset
    dataset_path = "data/emotions_risk_scores_2.csv"

    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return

    print("Dataset cargado correctamente")
    print(f"Registros: {len(df)}")

    # Inicializar modelo
    svm_model = SVMText()

    # =====================================================
    # VALIDACIÓN CRUZADA
    # =====================================================
    print("\nIniciando validación cruzada...")
    svm_model.validacion_cruzada(df, folds=10)

    # Guardar gráfica de CV
    svm_model.grafica_validacion_cruzada()

    # =====================================================
    # ENTRENAMIENTO
    # =====================================================
    #print("\nEntrenando modelo...")
    #svm_model.entrenamientoModelo(df)

    # =====================================================
    # EVALUACIÓN
    # =====================================================
    #print("\nEvaluando modelo...")
    #acc, report, cm = svm_model.evaluacion()

    # =====================================================
    # GUARDAR MODELO
    # =====================================================
    #print("\nGuardando modelo...")
    #svm_model.guardar_modelo("models/svm_text_model.pkl")

    print("\nProceso finalizado correctamente")


if __name__ == "__main__":
    main()