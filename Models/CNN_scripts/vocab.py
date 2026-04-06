from collections import Counter


def build_vocab(texts, max_size=5000):

    counter = Counter()

    for tokens in texts:
        counter.update(tokens)

    most_common = counter.most_common(max_size - 1)

    vocab = {"<PAD>": 0}

    for i, (word, _) in enumerate(most_common, start=1):
        vocab[word] = i

    return vocab