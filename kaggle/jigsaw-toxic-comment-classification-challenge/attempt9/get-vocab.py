from collections import defaultdict

import pandas as pd
from numpy import log

from common import *

ATTEMPT = '9'
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


def get_subvocab(data, df, wfilter, vid, n, max_size):
    idf = lambda w: log(len(data) / df[w])
    is_ngram = lambda w: w in wfilter and (w.count(' ') + 1) == n
    vocab = [(vid, w, df[w], idf(w)) for w in df if is_ngram(w)]
    vocab = sorted(vocab, key=lambda t: t[2], reverse=True)
    vocab = vocab[:max_size]
    return sorted(vocab, key=lambda t: t[1])


def get_vocab(data, progress=1000):
    df, _, normal_words, toxic_words = calc_df(data, progress)
    bad_words = toxic_words - normal_words
    good_words = normal_words - toxic_words
    bad_vocab = []
    good_vocab = []
    for n, max_size in MAX_VOCAB.items():
        bad_vocab += get_subvocab(data, df, bad_words, BAD_VOCAB, n, max_size)
        good_vocab += get_subvocab(data, df, good_words, GOOD_VOCAB, n,
                                   max_size)
    vocab = bad_vocab + good_vocab
    logmsg('total words=%d, good words=%d, bad words=%d',
           len(df), len(good_words), len(bad_words))
    return pd.DataFrame(data=vocab, columns=VOCAB_COLS)


if __name__ == '__main__':
    logmsg('Reading data ...')
    train_data = pd.read_csv(TRAIN_FILE, dtype={COMMENT: str})
    logmsg('Computing vocabulary ...')
    vocab = get_vocab(train_data)
    logmsg('Saving vocabulary (top %d n-grams) ...', len(vocab))
    vocab.to_csv(VOCAB_FILE, index=False)
