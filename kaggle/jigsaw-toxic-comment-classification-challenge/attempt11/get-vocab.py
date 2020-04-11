from collections import defaultdict

import pandas as pd

from common import *

ATTEMPT = 11
TRAIN_FILE = '../data/attempt%d/pre-train.csv' % ATTEMPT
VOCAB_FILE = '../data/attempt%d/vocab-%s.csv' % (ATTEMPT, ATTEMPT)


def get_vocab(data, progress=5000):
    vocab = defaultdict(lambda: 0)
    for i, row in data.iterrows():
        for word in get_row_words(row):
            vocab[word] += 1
        if (i + 1) % progress == 0:
            logmsg('Processed %d rows so far ...', i + 1)
    vocab = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
    vocab = ((i, w, f) for (i, (w, f)) in enumerate(vocab))
    return pd.DataFrame(data=vocab, columns=VOCAB_COLS)


if __name__ == '__main__':
    logmsg('Reading data ...')
    train_data = pd.read_csv(TRAIN_FILE, dtype={COMMENT: str})
    logmsg('Computing vocabulary ...')
    vocab = get_vocab(train_data)
    logmsg('Saving vocabulary (%d words) ...', len(vocab))
    vocab.to_csv(VOCAB_FILE, index=False)
