import pandas as pd
from NLP.pipeline import SentimentPipeline


def load_dataset(csv_path):

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    TEXT_COL = "original_text"
    LABEL_COL = "riesgo"

    pipeline = SentimentPipeline()

    texts = df[TEXT_COL].astype(str).tolist()
    labels_raw = df[LABEL_COL].astype(str).tolist()

    tokenized = []

    for text in texts:
        result = pipeline.process(text)
        tokenized.append(result["tokens"])

    label_map = {"bajo": 0, "medio": 1, "alto": 2}
    labels = [label_map[l.lower()] for l in labels_raw]

    return tokenized, labels