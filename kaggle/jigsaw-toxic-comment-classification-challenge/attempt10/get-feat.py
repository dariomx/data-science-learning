from collections import defaultdict
from os.path import join

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

from common import *

ATTEMPT = '9'
PREFIX = '../data/attempt%s' % ATTEMPT
VOCAB_FILE = join(PREFIX, 'vocab-%s.csv' % ATTEMPT)
TRAIN_FILE = join(PREFIX, 'pre-train.csv')
TRAIN_X_FILE = join(PREFIX, 'train-x.npz')
TRAIN_Y_FILE = join(PREFIX, 'train-y.npy')
TEST_FILE = join(PREFIX, 'pre-test.csv')
TEST_X_FILE = join(PREFIX, 'test-x.npz')


def get_vocab(vocab_file):
    vocab = pd.read_csv(vocab_file, usecols=[NGRAM], dtype={NGRAM: str})
    return {v: i for (i, v) in enumerate(vocab[NGRAM].values)}


def read_train_data(train_file):
    data = pd.read_csv(train_file, dtype={COMMENT: str})
    data[NONTOXIC] = (data[CATLAB].sum(axis=1) == 0).astype(int)
    return data


def calc_row_feat(row, vocab):
    row_feat = defaultdict(lambda: 0)
    row_words = get_row_ngrams(row)
    for w in row_words:
        i = vocab.get(w, -1)
        if i >= 0:
            row_feat[vocab[w]] = 1
    return row_feat


def calc_train_feat(data, vocab, progress=1000):
    x = []
    y = []
    used = set()
    for i, row in data.iterrows():
        row_feat = calc_row_feat(row, vocab)
        if len(row_feat) == 0:
            continue
        for cat in EXT_CATLAB:
            if row[cat] == 0:
                continue
            used.add(i)
            x.append(row_feat)
            y.append(CATIDX[cat])
        if (i + 1) % progress == 0:
            logmsg('Got training features for %d inputs ...' % (i + 1))
    logmsg('Generated %d x %d training matrix from %d/%d inputs...' %
           (len(x), len(vocab), len(used), len(data)))
    return x, y


def calc_test_feat(data, vocab, progress=1000):
    x = []
    for i, row in data.iterrows():
        row_feat = calc_row_feat(row, vocab)
        x.append(row_feat)
        if (i + 1) % progress == 0:
            logmsg('Got testing features for %d inputs ...' % (i + 1))
    logmsg('Generated %d testing rows from %d inputs...' %
           (len(x), len(data)))
    return x


# converts a list of dictionaries into a coo matrix; useful to persist in sparse
# format and recover later
def get_sparse_feat(x, vocab):
    row, col, val = [], [], []
    for i, feat in enumerate(x):
        for j, v in feat.items():
            row.append(i)
            col.append(j)
            val.append(v)
    shape = (len(x), len(vocab))
    return coo_matrix((val, (row, col)), shape=shape, dtype=np.float64)


def save_sparse_feat(x, vocab, fname):
    sx = get_sparse_feat(x, vocab)
    save_npz(fname, sx)


def save_dense_feat(x, fname):
    dy = np.array(x)
    np.save(fname, dy)


def get_train_feat(vocab_file, train_file, train_x_file, train_y_file):
    logmsg('Reading vocab ...')
    vocab = get_vocab(vocab_file)
    logmsg('Reading training data ...')
    train_data = read_train_data(train_file)
    logmsg('Extracting features from training data ...')
    x, y = calc_train_feat(train_data, vocab)
    logmsg('Saving training features ...')
    save_sparse_feat(x, vocab, train_x_file)
    save_dense_feat(y, train_y_file)
    return vocab


def get_test_feat(vocab, test_file, test_x_file):
    logmsg('Reading testing data ...')
    test_data = pd.read_csv(test_file, dtype={COMMENT: str})
    logmsg('Extracting features from testing data ...')
    x = calc_test_feat(test_data, vocab)
    logmsg('Saving testing features ...')
    save_sparse_feat(x, vocab, test_x_file)


if __name__ == '__main__':
    vocab = get_train_feat(VOCAB_FILE, TRAIN_FILE,
                           TRAIN_X_FILE, TRAIN_Y_FILE)
    get_test_feat(vocab, TEST_FILE, TEST_X_FILE)
