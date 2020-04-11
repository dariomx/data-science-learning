from os.path import join

import sys
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.svm import SVC

from common import *

ATTEMPT = 11
PREFIX = '../data/attempt%s' % ATTEMPT
TRAIN_X_FILE = join(PREFIX, 'train-x.npz')
TRAIN_Y_FILE = join(PREFIX, 'train-y.npy')
TEST_FILE = join(PREFIX, 'pre-test.csv')
TEST_X_FILE = join(PREFIX, 'test-x.npz')
PRED_FILE = join(PREFIX, 'pred-%d.csv' % ATTEMPT)


# convert to csr format to speed up computations (scipy recommendation?)
def load_sparse_mat(fname):
    return load_npz(fname).tocsr()


def train(x, y, C):
    logmsg('Will use SVC with C=%f', C)
    svc = SVC(probability=True, C=C, class_weight='balanced')
    svc.fit(x, y)
    return svc


def do_train(train_x_file, train_y_file, C):
    logmsg('Loading training data ...')
    x = load_sparse_mat(train_x_file)
    y = np.load(train_y_file)
    logmsg('Training the model ...')
    return train(x, y, C)


def test(model, x):
    return model.predict_proba(x)


def format_pred(prob, data):
    pred = []
    for i, in_row in data.iterrows():
        id = in_row[ID]
        out_row = {ID: id}
        for j, cat in enumerate(CATLAB):
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
    C = float(sys.argv[1])
    model = do_train(TRAIN_X_FILE, TRAIN_Y_FILE, C)
    do_test(model, TEST_FILE, TEST_X_FILE, PRED_FILE)
