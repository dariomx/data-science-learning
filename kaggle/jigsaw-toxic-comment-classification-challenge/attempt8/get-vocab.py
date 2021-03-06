from collections import defaultdict

import pandas as pd
from numpy import log

from common import *

ATTEMPT = '8'
TRAIN_FILE = '../data/attempt%s/pre-train.csv' % ATTEMPT
VOCAB_FILE = '../data/attempt%s/vocab-%s.csv' % (ATTEMPT, ATTEMPT)


def calc_df(data, progress):
    normal_words = set()
    toxic_words = set()
    df = defaultdict(lambda: 0)
    ntoxic_rows = 0
    for i, row in data.iterrows():
        is_toxic = (row[CATLAB].sum() > 0)
        ntoxic_rows += int(is_toxic)
        for w in set(get_row_ngrams(row)):
            df[w] += 1
            if is_toxic:
                toxic_words.add(w)
            else:
                normal_words.add(w)
        if (i + 1) % progress == 0:
            logmsg('Processed n-grams from %d input rows ...', i + 1)
    return df, ntoxic_rows, normal_words, toxic_words


def get_vocab(data, progress=1000):
    df, ntoxic_rows, normal_words, toxic_words = calc_df(data, progress)
    bad_words = toxic_words - normal_words
    idf = lambda w: log(len(data) / df[w])
    vocab = [(w, df[w], idf(w)) for w in df if w in bad_words]
    vocab = sorted(vocab, key=lambda t: t[1], reverse=True)
    vocab = vocab[:MAX_VOCAB]
    vocab = sorted(vocab, key=lambda t: t[0])
    logmsg("normal rows=%d, toxic rows=%d", len(data) - ntoxic_rows,
           ntoxic_rows)
    logmsg("normal words=%d, toxic words=%d, bad words=%d",
           len(df) - len(toxic_words), len(toxic_words), len(bad_words))
    return pd.DataFrame(data=vocab, columns=VOCAB_COLS)


if __name__ == '__main__':
    logmsg('Reading data ...')
    train_data = pd.read_csv(TRAIN_FILE, dtype={COMMENT: str})
    logmsg('Computing vocabulary ...')
    vocab = get_vocab(train_data)
    logmsg('Saving vocabulary (top %d bad words) ...', len(vocab))
    vocab.to_csv(VOCAB_FILE, index=False)
