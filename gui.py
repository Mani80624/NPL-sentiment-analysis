import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from NLP.Plutchik_visualization import Visualization_Plutchik
from NLP.risk_pipeline import RiskDetectionPipeline
from tools.model_manager import ModelManager


from gui.estilos import (
    COLOR_FONDO_VENTANA,
    COLOR_PANEL_IZQUIERDO,
    COLOR_PANEL_DERECHO,
    COLOR_TEXTO_CLARO,
    COLOR_SUBTITULO,
    COLOR_SLIDER_FONDO,
    COLOR_SLIDER_BARRA,
    COLOR_SLIDER_ACTIVO,
    FUENTE_SUBTITULO,
    FUENTE_PEQUENA
)

from gui.componentes import (
    crear_titulo,
    crear_boton,
    crear_texto_scroll,
    crear_panel
)

class NLP_GUI:

    def __init__(self, root):
        self.root = root
        self.root.title("NLP Risk Detector")
        self.root.geometry("1400x750")
        self.root.configure(bg=COLOR_FONDO_VENTANA)
        
        self.pipeline = RiskDetectionPipeline()
        self.model_manager = ModelManager()
        self.visualization = Visualization_Plutchik()
        
        self.result = None
        self.features = None
        
        self.main_frame = tk.Frame(
            self.root,
            bg=COLOR_FONDO_VENTANA)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.left_frame = crear_panel(self.main_frame, COLOR_PANEL_IZQUIERDO)
        self.left_frame.pack(side="left", fill="both", expand=True)
        
        self.right_frame = crear_panel(self.main_frame, COLOR_PANEL_DERECHO)
        self.right_frame.pack(side="right", fill="both", expand=True)
        
        self.panel_izquierdo()
        self.panel_derecho()
        
        # Interfaz
        
    def panel_izquierdo(self):
        crear_titulo(self.left_frame, "Text").pack(pady=(12, 8))
        self.input = crear_texto_scroll(self.left_frame, alto=8, ancho=80)
        self.input.pack(fill="x", padx=12, pady=(0, 10))
            
        self.boton_procesar = crear_boton(
        self.left_frame,
        texto="Process Text",
        comando=self.procesar,
        tipo="primario"
        )
        self.boton_procesar.pack(pady=(0, 8))

        self.boton_limpiar = crear_boton(
            self.left_frame,
            texto="Clean",
            comando=self.limpiar,
            tipo="limpiar"
        )
        self.boton_limpiar.pack(pady=(0, 12))
        
        tk.Label(
            self.left_frame,
            text="Risk models",
            font=FUENTE_SUBTITULO,
            bg=COLOR_PANEL_IZQUIERDO,
            fg=COLOR_SUBTITULO
        ).pack()

        self.modelo = tk.IntVar(value=1)
        
        tk.Label(
            self.left_frame,
            text="1: Naive Bayes   2: CNN   3: Random Forest   4: SVM",
            font=FUENTE_PEQUENA,
            bg=COLOR_PANEL_IZQUIERDO,
            fg=COLOR_TEXTO_CLARO
        ).pack(pady=6)
        
        self.slider = tk.Scale(
            self.left_frame,
            from_=1,
            to=4,
            orient=tk.HORIZONTAL,
            variable=self.modelo,
            command=self.actualizar_modelo,
            length=320,
            showvalue=True,
            bg=COLOR_SLIDER_FONDO,
            fg=COLOR_TEXTO_CLARO,
            troughcolor=COLOR_SLIDER_BARRA,
            activebackground=COLOR_SLIDER_ACTIVO,
            highlightthickness=0,
            bd=0
        )
        self.slider.pack(pady=(0, 12))
        
        self.output = crear_texto_scroll(self.left_frame, alto=20, ancho=90)
        self.output.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
    # Panel Derecho
    def panel_derecho(self):
        crear_titulo(self.right_frame, "Plutchik's Graph").pack(pady=(12, 8))
            
        self.plot_canvas = FigureCanvasTkAgg(
        self.visualization.get_figure(),
        master=self.right_frame)
            
        self.plot_canvas.get_tk_widget().pack(
        fill="both",
        expand=True,
        padx=12,
        pady=(0, 12))
        
    # -------- lógica --------

    def procesar(self):
        texto = self.input.get("1.0", tk.END).strip()

        if not texto:
            return
        self.result = self.pipeline.process(texto)
        self.features = self.result["risk_features"]

        self.mostrar_resultado()
        self.actualizar_plot()
        
    def limpiar(self):
        self.input.delete("1.0", tk.END)
        self.output.delete("1.0", tk.END)
        
        self.result = None
        self.features = None
        
        self.modelo.set(1)
        
        self.visualization.draw({
            "joy_count": 0,
            "trust_count": 0,
            "fear_count": 0,
            "surprise_count": 0,
            "sadness_count": 0,
            "disgust_count": 0,
            "anger_count": 0,
            "anticipation_count": 0,
            },
            "No Data"
            )
        self.plot_canvas.draw()

    def actualizar_modelo(self, event=None):
        if self.features:
            self.mostrar_resultado()
            self.actualizar_plot()
            
    def actualizar_plot (self):
        if not self.features:
            return
        nombres = {
            1: "Naive Bayes",
            2: "CNN",
            3: "Random Forest",
            4: "SVM"
        }
        modelo_actual = nombres[self.modelo.get()]
        self.visualization.draw(self.features, modelo_actual)
        self.plot_canvas.draw()

    def calcular_riesgo(self):
        #Agrega texto 
        modelo_id = self.modelo.get()
        texto_orginal = self.input.get("1.0", tk.END).strip()

        return self.model_manager.predict(
            modelo_id,
            self.features,
            raw_text = texto_orginal
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
            f"MODEL: {nombres[self.modelo.get()]}\n\n"
            f"DOMINANT EMOTION: {self.result['dominant_emotion']}\n\n"
            f"RISK:\n"
            f"  Level: {level}\n"
            f"  Score: {round(score,2)}\n\n"
            f"EMOTIONS DETECTED:\n"
            f"  Joy: {self.features.get('joy_count', 0)}\n"
            f"  Trust: {self.features.get('trust_count', 0)}\n"
            f"  Fear: {self.features.get('fear_count', 0)}\n"
            f"  Surprise: {abs(self.features.get('surprise_count', 0))}\n"
            f"  Sadness: {self.features.get('sadness_count', 0)}\n"
            f"  Disgust: {self.features.get('disgust_count', 0)}\n"
            f"  Anger: {self.features.get('anger_count', 0)}\n"
            f"  Anticipation: {abs(self.features.get('anticipation_count', 0))}\n"
            f"  Text length: {self.features.get('text_length', 0)}"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = NLP_GUI(root)
    root.mainloop()