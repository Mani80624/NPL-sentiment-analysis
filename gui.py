import tkinter as tk
from tkinter import scrolledtext

from NLP.risk_pipeline import RiskDetectionPipeline
from tools.model_manager import ModelManager


class NLP_GUI:

    def __init__(self, root):

        self.root = root

        self.pipeline = RiskDetectionPipeline()
        self.model_manager = ModelManager()

        self.result = None
        self.features = None

        root.title("Detector de Riesgo NLP")
        root.geometry("800x600")

        tk.Label(root, text="Texto", font=("Arial", 14)).pack()

        self.input = scrolledtext.ScrolledText(root, height=8, width=80)
        self.input.pack()

        tk.Button(
            root,
            text="Procesar Texto",
            command=self.procesar,
            bg="lightblue"
        ).pack(pady=10)

        # Slider modelos
        tk.Label(root, text="Modelo de riesgo").pack()

        self.modelo = tk.IntVar(value=1)

        self.slider = tk.Scale(
            root,
            from_=1,
            to=4,
            orient=tk.HORIZONTAL,
            variable=self.modelo,
            command=self.actualizar_modelo,
            label="1:Bayes  2:CNN  3:RF  4:SVM"
        )
        self.slider.pack()

        self.output = scrolledtext.ScrolledText(root, height=20, width=90)
        self.output.pack()

    # -------- lógica --------

    def procesar(self):
        texto = self.input.get("1.0", tk.END).strip()

        if not texto:
            return

        self.result = self.pipeline.process(texto)
        self.features = self.result["risk_features"]

        self.mostrar_resultado()

    def actualizar_modelo(self, event=None):
        if self.features:
            self.mostrar_resultado()

    def calcular_riesgo(self):

        modelo_id = self.modelo.get()

        return self.model_manager.predict(
            modelo_id,
            self.features
        )

    def mostrar_resultado(self):

        self.output.delete("1.0", tk.END)

        score, level = self.calcular_riesgo()

        nombres = {
            1: "Naive Bayes",
            2: "CNN",
            3: "Random Forest",
            4: "SVM"
        }

        self.output.insert(
            tk.END,
            f"MODELO: {nombres[self.modelo.get()]}\n\n"
            f"EMOCION DOMINANTE: {self.result['dominant_emotion']}\n\n"
            f"RIESGO:\nNivel: {level}\nScore: {score}\n\n"
            f"EMOCIONES:\n{self.features}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = NLP_GUI(root)
    root.mainloop()