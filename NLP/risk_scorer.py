class RiskScorer:
    """
    Condiciones para ver el riesgo de suicidio dando diferentes pesos a las clases
    """
    def score(self, features):
        score = (
            features["hopeless_count"] * 2 +
            features["death_count"] * 5 +
            features["goodbye_count"] * 4
        )

        if score >= 10:
            level = "ALTO"
        elif score >= 5:
            level = "MEDIO"
        else:
            level = "BAJO"

        return score, level