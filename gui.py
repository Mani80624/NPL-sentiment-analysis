import tkinter as tk
from tkinter import scrolledtext

from NLP.normalizacion import TextNormalizer
from NLP.tokenizer import Tokenizer
from NLP.stopwords import StopWordsRemover
from NLP.sentiment_analyzer import SentimentAnalyzer
from NLP.risk_pipeline import RiskDetectionPipeline


class NLP_GUI:

    def __init__(self, root):

        self.root = root # Ventana

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
        self.diccionario = {'clean_text': [], 'tokens':[],
                            'tokens_stopwords':[], 
                            'emotion':[], 'counter':[], 
                            'risk_score':[],'risk_level':[], 'risk_feature':[]}
        

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

    def procesamiento(self):
        texto = self.input.get("1.0", tk.END)
        self.diccionario = self.risk.process(texto)
    

    def limpiar_salida(self):
        self.output.delete("1.0", tk.END)

    def normalizar(self):
        self.limpiar_salida()
        self.procesamiento()
        self.output.insert(tk.END, f"TEXTO NORMALIZADO:\n{self.diccionario['clean_text']}")

    def tokenizar(self):

        self.limpiar_salida()
        if not self.diccionario['tokens']:
            self.procesamiento()
        self.output.insert(tk.END, f"TOKENS:\n{self.diccionario['tokens']}")

    def filtrar(self):

        self.limpiar_salida()
        if not self.diccionario['tokens_stopwords']:
            self.procesamiento()

        self.output.insert(tk.END, f"TOKENS FILTRADOS:\n{self.diccionario['tokens_stopwords']}")

    def sentimiento(self):

        self.limpiar_salida()

        if not self.diccionario['emotion'] and not self.diccionario['counter']:
            self.procesamiento()

        self.output.insert(tk.END, f"EMOCION:\n{self.diccionario['emotion']}\n\nCONTEO:\n{self.diccionario['counter']}")

    def riesgo(self):

        self.limpiar_salida()
        if not self.diccionario['risk_level'] and not self.diccionario['risk_score']:
            self.procesamiento()

        self.output.insert(
            tk.END,
            f"RIESGO:\nNivel: {self.diccionario['risk_level']}\nScore: {self.diccionario['risk_score']}"
        )

    def resetear(self):

        self.clean_text = None
        self.tokens = None
        self.filtered_tokens = None
        self.diccionario = {'clean_text': [], 'tokens':[], 
                            'tokens_stopwords':[],
                            'emotion':[], 'counter':[], 
                            'risk_score':[],'risk_level':[], 'risk_feature':[]}

        self.input.delete("1.0", tk.END)
        self.limpiar_salida()


if __name__ == "__main__":
    root = tk.Tk()
    app = NLP_GUI(root)
    root.mainloop()