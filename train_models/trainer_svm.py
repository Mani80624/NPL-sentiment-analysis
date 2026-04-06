from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, df, text_col, label_col):
        X = df[text_col]
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        self.model.train(X_train, y_train)

        return X_test, y_test