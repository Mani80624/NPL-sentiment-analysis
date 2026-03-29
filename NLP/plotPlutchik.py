from pyplutchik import plutchik
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Plot:
    def plot(self, diccionario):
        emociones_base = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]
        for emo in emociones_base:
            if emo not in diccionario:
                diccionario[emo] = 0.0

        plutchik(diccionario, show_coordinates=True, show_ticklabels=True)
        plt.show()
