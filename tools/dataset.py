import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, tokenized_texts, labels, vocab, max_len=50):

        self.texts = tokenized_texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, tokens):

        indices = [self.vocab.get(t, 0) for t in tokens]

        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        x = self.encode(self.texts[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y