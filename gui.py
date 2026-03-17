import tkinter as tk
from tkinter import scrolledtext

from NLP.normalizacion import TextNormalizer
from NLP.tokenizer import Tokenizer
from NLP.stopwords import StopWordsRemover
from NLP.sentiment_analyzer import SentimentAnalyzer
from NLP.risk_pipeline import RiskDetectionPipeline


class NLP_GUI:

    def __init__(self, root):

        self.root = root

        # módulos NLP
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        self.stopwords = StopWordsRemover()
        self.sentiment = SentimentAnalyzer()
        self.risk = RiskDetectionPipeline()

        # ESTADO
        self.clean_text = None
        self.tokens = None
        self.filtered_tokens = None

        root.title("Laboratorio NLP")
        root.geometry("800x700")

        tk.Label(root, text="Texto", font=("Arial", 14)).pack()

        self.input = scrolledtext.ScrolledText(root, height=8, width=80)
        self.input.pack()

        frame = tk.Frame(root)
        frame.pack(pady=5)

        tk.Button(frame, text="Normalizar", command=self.normalizar).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="Tokenizar", command=self.tokenizar).grid(row=0, column=1, padx=5)
        tk.Button(frame, text="StopWords", command=self.filtrar).grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Sentiment", command=self.sentimiento).grid(row=0, column=3, padx=5)
        tk.Button(frame, text="Riesgo", command=self.riesgo).grid(row=0, column=4, padx=5)

        tk.Button(root, text="Limpiar Todo", command=self.resetear, bg="orange").pack()

        self.output = scrolledtext.ScrolledText(root, height=20, width=90)
        self.output.pack()

    # -------- funciones ----------

    def limpiar_salida(self):
        self.output.delete("1.0", tk.END)

    def normalizar(self):

        self.limpiar_salida()

        texto = self.input.get("1.0", tk.END)

        self.clean_text, _ = self.normalizer.normalize(texto)

        self.output.insert(tk.END, f"TEXTO NORMALIZADO:\n{self.clean_text}")

    def tokenizar(self):

        self.limpiar_salida()

        if not self.clean_text:
            self.normalizar()

        self.tokens = self.tokenizer.tokenize(self.clean_text)

        self.output.insert(tk.END, f"TOKENS:\n{self.tokens}")

    def filtrar(self):

        self.limpiar_salida()

        if not self.tokens:
            self.tokenizar()

        self.filtered_tokens = self.stopwords.remove(self.tokens)

        self.output.insert(tk.END, f"TOKENS FILTRADOS:\n{self.filtered_tokens}")

    def sentimiento(self):

        self.limpiar_salida()

        if not self.filtered_tokens:
            self.filtrar()

        emotion, counter = self.sentiment.analyze(self.filtered_tokens)

        self.output.insert(tk.END, f"EMOCION:\n{emotion}\n\nCONTEO:\n{counter}")

    def riesgo(self):

        self.limpiar_salida()

        texto = self.input.get("1.0", tk.END)

        result = self.risk.process(texto)

        self.output.insert(
            tk.END,
            f"RIESGO:\nNivel: {result['risk_level']}\nScore: {result['risk_score']}"
        )

    def resetear(self):

        self.clean_text = None
        self.tokens = None
        self.filtered_tokens = None

        self.input.delete("1.0", tk.END)
        self.limpiar_salida()


if __name__ == "__main__":
    root = tk.Tk()
    app = NLP_GUI(root)
    root.mainloop()