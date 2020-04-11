from collections import defaultdict
from os.path import join
from time import process_time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

from common import *

ATTEMPT = 11
PREFIX = '../data/attempt%d' % ATTEMPT
VOCAB_FILE = join(PREFIX, 'vocab-%d.csv' % ATTEMPT)
TRAIN_FILE = join(PREFIX, 'pre-train.csv')
TRAIN_X_FILE = join(PREFIX, 'train-x.npy')
TRAIN_Y_FILE = join(PREFIX, 'train-y.npy')
TEST_FILE = join(PREFIX, 'pre-test.csv')
TEST_X_FILE = join(PREFIX, 'test-x.npy')


# the last two flags aim to avoid that "null" gets parsed as NA
def get_vocab(vocab_file):
    vocab = pd.read_csv(vocab_file,
                        usecols=[WID, WORD],
                        dtype={WORD: str},
                        na_values=None,
                        keep_default_na=False)
    return {v: i for (i, v) in vocab.itertuples(index=False, name=None)}


def read_data(fname, train=True):
    data = pd.read_csv(fname,
                       dtype={COMMENT: str},
                       na_values=None,
                       keep_default_na=False)
    if train:
        data[NONTOXIC] = (data[CATLAB].sum(axis=1) == 0).astype(int)
    return data


def calc_row_feat(row, vocab):
    row_feat = defaultdict(lambda: 0)
    row_words = get_row_words(row)
    for w in row_words:
        i = vocab.get(w, -1)
        if i >= 0:
            row_feat[vocab[w]] = 1
    return row_feat


# converts a list of dictionaries (one-hot encodings) into a sparse csr matrix
def get_sparse_feat(x, vocab):
    row, col, val = [], [], []
    for i, feat in enumerate(x):
        for j, v in feat.items():
            row.append(i)
            col.append(j)
            val.append(v)
    shape = (len(x), len(vocab))
    coo = coo_matrix((val, (row, col)), shape=shape, dtype=np.float64)
    return coo.tocsr()


def calc_train_feat(data, vocab, progress=5000):
    x = []
    y = []
    for i, row in data.iterrows():
        row_feat = calc_row_feat(row, vocab)
        if len(row_feat) == 0:
            continue
        for cat in EXT_CATLAB:
            if row[cat] == 0:
                continue
            x.append(row_feat)
            y.append(CATIDX[cat])
        if (i + 1) % progress == 0:
            logmsg('Got training features for %d inputs ...' % (i + 1))
    logmsg('Generated %d x %d training matrix from %d inputs...' %
           (len(x), len(vocab), len(data)))
    return get_sparse_feat(x, vocab), y


def calc_test_feat(data, vocab, progress=5000):
    x = []
    for i, row in data.iterrows():
        row_feat = calc_row_feat(row, vocab)
        x.append(row_feat)
        if (i + 1) % progress == 0:
            logmsg('Got testing features for %d inputs ...' % (i + 1))
    logmsg('Generated %d testing rows from %d inputs...' %
           (len(x), len(data)))
    return get_sparse_feat(x, vocab)


def reduce_dim(x, k=2000, n_iter=10):
    start = process_time()
    svd = TruncatedSVD(n_components=k, n_iter=n_iter)
    x_k = svd.fit_transform(x)
    end = process_time()
    logmsg('Truncated SVD took %f secs', end - start)
    logmsg('Reduced to %d x %d' % x_k.shape)
    logmsg('Explained variance = %f' % sum(svd.explained_variance_ratio_))
    return x_k, lambda y: svd.transform(y)


def save_dense_feat(x, fname):
    dy = np.array(x)
    np.save(fname, dy)


def get_train_feat(vocab_file, train_file, train_x_file, train_y_file):
    logmsg('Reading vocab ...')
    vocab = get_vocab(vocab_file)
    logmsg('Reading training data ...')
    train_data = read_data(train_file)
    logmsg('Extracting features from training data ...')
    x, y = calc_train_feat(train_data, vocab)
    logmsg('Reducing dimension of training features ...')
    x_k, reducer = reduce_dim(x)
    logmsg('Saving training features ...')
    save_dense_feat(x_k, train_x_file)
    save_dense_feat(y, train_y_file)
    return vocab, reducer


def get_test_feat(vocab, reducer, test_file, test_x_file):
    logmsg('Reading testing data ...')
    test_data = read_data(test_file, train=False)
    logmsg('Extracting features from testing data ...')
    x = calc_test_feat(test_data, vocab)
    logmsg('Reducing dimension of testing features ...')
    x_k = reducer(x)
    logmsg('Saving testing features ...')
    save_dense_feat(x_k, test_x_file)


if __name__ == '__main__':
    vocab, reducer = get_train_feat(VOCAB_FILE, TRAIN_FILE,
                                    TRAIN_X_FILE, TRAIN_Y_FILE)
    get_test_feat(vocab, reducer, TEST_FILE, TEST_X_FILE)
