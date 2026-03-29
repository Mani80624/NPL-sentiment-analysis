class RiskScorer:

    def score(self, features):
        """
        Pesos de cada emoción de acuerdo con lo dado por la
        psiquiatra
        sadness -> 30%
        fear -> 20%
        anger -> 16%
        disgust -> 6%
        trust -> 12%
        surprice-> 8%
        anticipation -> 5%
        joy -> 3%
        """
        if features['joy_count'] and features['fear_count']:
            features['anticipation_count'] = -features['anticipation_count']
        if features['joy_count'] and features['trust_count']:
            features["surprise_count"] = -features["surprise_count"]

        score = (
            features["sadness_count"] * 0.3 +
            features["fear_count"] * 0.2 +
            features["anger_count"] * 0.16 +
            features["disgust_count"] * 0.06 -
            features["joy_count"] * 0.03 -
            features["trust_count"] * 0.12+
            features["anticipation_count"]*0.05+
            features["surprise_count"]*0.08
        )
        

        return score
    
    
