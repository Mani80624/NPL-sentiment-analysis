import pandas as pd
from NLP.risk_scorer import RiskScorer
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats.mstats import winsorize

# Load objects
risk_scorer = RiskScorer()
robust_scaler = RobustScaler()

# Files Import
df_1 = pd.read_csv("C:/Users/ma-nu/Downloads/minería de datos/counts_leghts_chavos.csv")
df_2 = pd.read_csv("C:/Users/ma-nu/Downloads/minería de datos/counts_leghts_ochoa.csv")
df = pd.concat([df_1, df_2], axis=0, ignore_index=False)
df["score"] = df.apply(risk_scorer.score, axis=1)

# Plot distibution scores
def plots(df, name):
    scores = df[name]
    plt.figure(figsize=(14,6))

    # Histograma
    plt.subplot(1,2,1)
    sns.histplot(scores, bins=50, kde=True, color="steelblue")
    plt.axvline(scores.median(), color="green", linestyle="--", label="Mediana")
    plt.axvline(scores.quantile(0.75), color="orange", linestyle="--", label="p75")
    plt.title("Distribución de Scores")
    plt.xlabel("Score")
    plt.ylabel("Frecuencia")
    plt.legend()

    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=scores, color="lightblue")
    plt.title("Boxplot de Scores")
    plt.xlabel("Score")

    plt.tight_layout()
    plt.show()


# Normalitation with lenght texts
def normalizar_logitud_datos(df):
    df["score_norm"] = round(df["score"]/df["text_length"],4)
    return df


norm_score = normalizar_logitud_datos(df.reset_index(drop=True))
"""
x <= -0.46 Bajo
-0.46 > x <=0.53 Mediano
x > 0.53 riesgo alto
"""
# Supongamos que tu DataFrame se llama df y la columna es "robustS"
def risk_level(df):
    df.loc[df["score_wins"] <= 0.0, "riesgo"] = "Bajo"
    
    df.loc[(df["score_wins"] > 0.0) & (df["score_wins"] <= 0.035), "riesgo"] = "Medio"
    
    df.loc[df["score_wins"] >= 0.035, "riesgo"] = "Alto"

    return df

scores = norm_score['score_norm'].values
scores_wins = winsorize(scores, limits=[0.04, 0.04])
norm_score["score_wins"] = scores_wins
norm_score = risk_level(norm_score)

norm_score.to_csv('emotions_risk_scores_1.csv', index=False)

