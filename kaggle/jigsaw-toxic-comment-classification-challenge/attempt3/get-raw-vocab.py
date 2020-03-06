from collections import defaultdict

import pandas as pd

from common import *

TRAIN_DATA = '../data/attempt3/pre-train.csv'
RAW_VOCAB = '../data/attempt3/raw-vocab.txt'
MAX_VOCAB = 2000


def get_raw_vocab(data):
    vocab = defaultdict(lambda: 0)
    toxic_data = data[data[CATLAB].sum(axis=1) > 0]
    for _, row in toxic_data.iterrows():
        for w in get_row_words(row):
            vocab[w] += 1
    vocab = sorted(vocab.items(), key=lambda t: -t[1])
    return [w for (w, _) in vocab[:MAX_VOCAB]]


if __name__ == '__main__':
    print('Reading data ...')
    train_data = pd.read_csv(TRAIN_DATA, dtype={COMMENT: str})
    raw_vocab = get_raw_vocab(train_data)
    with open(RAW_VOCAB, 'w') as f:
        f.write('\n'.join(raw_vocab))
