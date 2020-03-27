from os.path import join

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.svm import SVC

from common import *

ATTEMPT = '8'
PREFIX = '../data/attempt%s' % ATTEMPT
TRAIN_X_FILE = join(PREFIX, 'train-x.npz')
TRAIN_Y_FILE = join(PREFIX, 'train-y.npy')
TEST_FILE = join(PREFIX, 'pre-test.csv')
TEST_X_FILE = join(PREFIX, 'test-x.npz')
PRED_FILE = join(PREFIX, 'pred-%s.csv' % ATTEMPT)


# convert to csr format to speed up computations (scipy recommendation?)
def load_sparse_mat(fname):
    return load_npz(fname).tocsr()


def train(x, y):
    svc = SVC(probability=True)
    svc.fit(x, y)
    return svc


def do_train(train_x_file, train_y_file):
    logmsg('Loading training data ...')
    x = load_sparse_mat(train_x_file)
    y = np.load(train_y_file)
    logmsg('Training the model ...')
    return train(x, y)


def test(model, x):
    return model.predict_proba(x)


def format_pred(prob, data):
    pred = []
    for i, in_row in data.iterrows():
        id = in_row[ID]
        out_row = {ID: id}
        for cat, j in CATIDX.items():
            out_row[cat] = prob[i, j]
        pred.append(out_row)
    return pd.DataFrame(pred, columns=COLS)


def do_test(model, test_file, test_x_file, pred_file):
    logmsg('Loading testing data ...')
    data = pd.read_csv(test_file, usecols=[ID], dtype={COMMENT: str})
    x = load_sparse_mat(test_x_file)
    logmsg('Testing the model ...')
    prob = test(model, x)
    logmsg('Saving predictions ...')
    pred = format_pred(prob, data)
    pred.to_csv(pred_file, index=False)


if __name__ == '__main__':
    model = do_train(TRAIN_X_FILE, TRAIN_Y_FILE)
    do_test(model, TEST_FILE, TEST_X_FILE, PRED_FILE)
