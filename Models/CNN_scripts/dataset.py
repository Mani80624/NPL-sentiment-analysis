import torch


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, texts, labels, vocab, max_len=150):
        """
        texts: lista de textos (str) o lista de tokens (list[str])
        labels: lista de etiquetas (int)
        vocab: diccionario palabra → índice
        max_len: longitud máxima de secuencia
        """

        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        """
        Convierte texto o lista de tokens a índices
        """

        # Detectar tipo de entrada
        if isinstance(text, list):
            tokens = text

        elif isinstance(text, str):
            tokens = text.split()

        else:
            # caso raro (None, NaN, etc)
            tokens = []

        # convertir a índices
        ids = [self.vocab.get(token, 0) for token in tokens]

        # padding / truncado
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.encode(text)

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )