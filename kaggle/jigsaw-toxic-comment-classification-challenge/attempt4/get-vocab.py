from collections import defaultdict
from itertools import chain

import pandas as pd

from common import *

TRAIN_DATA = '../data/attempt4/pre-train.csv'
VOCAB_DATA = '../data/attempt4/vocab.txt'


def get_vocab_from(data):
    vocab = defaultdict(lambda: defaultdict(lambda: 0))
    for _, row in data.iterrows():
        for w in get_row_ngrams(row):
            k = w.count(' ') + 1
            vocab[k][w] += 1
    for k, max_k in MAX_VOCAB.items():
        k_vocab = sorted(vocab[k].items(), key=lambda t: -t[1])
        k_vocab = [w for (w, _) in k_vocab[:max_k]]
        vocab[k] = k_vocab
    return list(chain.from_iterable(vocab.values()))


def get_vocab(data):
    normal_data = data[data[CATLAB].sum(axis=1) == 0]
    toxic_data = data[data[CATLAB].sum(axis=1) > 0]
    print("normal=%d, toxic=%d" % (len(normal_data), len(toxic_data)))
    normal_vocab = get_vocab_from(normal_data)
    toxic_vocab = get_vocab_from(toxic_data)
    return normal_vocab + toxic_vocab


if __name__ == '__main__':
    print('Reading data ...')
    train_data = pd.read_csv(TRAIN_DATA, dtype={COMMENT: str})
    vocab = get_vocab(train_data)
    with open(VOCAB_DATA, 'w') as f:
        f.write('\n'.join(vocab))
