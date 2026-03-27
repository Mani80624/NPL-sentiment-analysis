class RiskScorer:

    def score(self, features):

        score = (
            features["sadness_count"] * 3 +
            features["fear_count"] * 3 +
            features["anger_count"] * 2 +
            features["disgust_count"] * 2 -
            features["joy_count"] * 2 -
            features["trust_count"] * 2
        )

        if score >= 15:
            level = "ALTO"
        elif score >= 7:
            level = "MEDIO"
        else:
            level = "BAJO"

        return score, level