import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from pyplutchik import plutchik


class Visualization_Plutchik:
    def __init__(self):
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)

    def get_figure(self):
        return self.fig

    def preparar_emociones(self, features):
        emociones = {
            "joy": max(0, features.get("joy_count", 0)),
            "trust": max(0, features.get("trust_count", 0)),
            "fear": max(0, features.get("fear_count", 0)),
            "surprise": max(0, features.get("surprise_count", 0)),
            "sadness": max(0, features.get("sadness_count", 0)),
            "disgust": max(0, features.get("disgust_count", 0)),
            "anger": max(0, features.get("anger_count", 0)),
            "anticipation": max(0, features.get("anticipation_count", 0)),
        }
        
        max_val = max(emociones.values()) if emociones.values() else 1
        if max_val == 0:
            max_val = 1

        emociones_normalizadas = {
            emo: val / max_val
            for emo, val in emociones.items()
        }
        return emociones_normalizadas

    def draw(self, features, model_name=None):
        emociones = self.preparar_emociones(features)

        self.fig.clear()

        # Algunos releases de pyplutchik dibujan sobre el eje actual.
        ax = self.fig.add_subplot(111)
        
        plutchik(
            emociones,
            ax=ax,
            show_coordinates=True,
            show_ticklabels=True
        )
        
        titulo = "Plutchik"
        if model_name:
            titulo += f" - {model_name}"

        ax.set_title(titulo, pad=20)
        