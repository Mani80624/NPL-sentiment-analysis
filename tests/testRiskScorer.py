from NLP.risk_scorer import RiskScorer

riskscore = RiskScorer()
features = {
    'hopeless_count':2,
    'death_count' : 3,
    'goodbye_count':1
}

score, level = riskscore.score(features)

"""
Sugerencia:
Puede haber una escala de probabilidad que indique qué tanto afectan los factores
de hopeless count, death count y goodbye count. Ejemplo:
hopeless count : 50%
death count: 30%
goodbye count: 20%
"""

print(f'El riesgo es: {level}, con un score de: {score}')