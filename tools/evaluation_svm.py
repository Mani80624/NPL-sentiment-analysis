from sklearn.metrics import classification_report, accuracy_score

class Evaluator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("\nAccuracy:", acc)
        print("\nReporte:\n", report)

        return acc, report