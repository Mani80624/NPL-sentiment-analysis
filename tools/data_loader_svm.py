import pandas as pd

class DataLoader:
    def __init__(self, text_col="original_text", label_col="riesgo"):
        self.text_col = text_col
        self.label_col = label_col

    def load_data(self, path):
        df = pd.read_csv(path)

        df = df[[self.text_col, self.label_col]].copy()
        df[self.text_col] = df[self.text_col].fillna("").astype(str)
        df[self.label_col] = df[self.label_col].fillna("").astype(str)

        df = df[df[self.text_col].str.strip() != ""]
        df = df[df[self.label_col].str.strip() != ""]

        print(f"[DATA] Registros: {len(df)}")
        return df