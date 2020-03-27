from collections import defaultdict

import pandas as pd
from numpy import log

from common import *

ATTEMPT = '8'
TRAIN_FILE = '../data/attempt%s/pre-train.csv' % ATTEMPT
VOCAB_FILE = '../data/attempt%s/vocab-%s.csv' % (ATTEMPT, ATTEMPT)


# uses idf to sort n-grams by relevance
def get_vocab_from(data):
    n = len(data)
    df = defaultdict(lambda: 0)
    for i, row in data.iterrows():
        for w in set(get_row_ngrams(row)):
            df[w] += 1
    idf = lambda w: log(n / df[w])
    vocab = [(w, df[w], idf(w), df[w] * idf(w)) for w in df]
    vocab = sorted(vocab, key=lambda t: t[3], reverse=True)
    vocab = vocab[:MAX_VOCAB]
    return pd.DataFrame(data=vocab, columns=VOCAB_COLS)


def get_vocab(data):
    normal_data = data[data[CATLAB].sum(axis=1) == 0]
    toxic_data = data[data[CATLAB].sum(axis=1) > 0]
    print("normal=%d, toxic=%d" % (len(normal_data), len(toxic_data)))
    return get_vocab_from(toxic_data)


if __name__ == '__main__':
    print('Reading data ...')
    train_data = pd.read_csv(TRAIN_FILE, dtype={COMMENT: str})
    vocab = get_vocab(train_data)
    vocab.to_csv(VOCAB_FILE, index=False)
