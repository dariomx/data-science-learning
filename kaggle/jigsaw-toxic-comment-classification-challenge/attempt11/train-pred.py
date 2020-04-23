from os.path import join

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from common import *

ATTEMPT = 11
PREFIX = '../data/attempt%s' % ATTEMPT
TRAIN_X_FILE = join(PREFIX, 'train-x.npy')
TRAIN_Y_FILE = join(PREFIX, 'train-y.npy')
TEST_FILE = join(PREFIX, 'pre-test.csv')
TEST_X_FILE = join(PREFIX, 'test-x.npy')
PRED_FILE = join(PREFIX, 'pred-%d.csv' % ATTEMPT)


def train(x, y, C, gamma):
    logmsg('Will use SVC with C=%f', C)
    svc = SVC(probability=True, C=C, gamma=gamma, class_weight='balanced')
    svc.fit(x, y)
    return svc


def do_train(train_x_file, train_y_file, C, gamma):
    logmsg('Loading training data ...')
    x = np.load(train_x_file)
    y = np.load(train_y_file)
    logmsg('Training the model (%d x %d)...' % x.shape)
    return train(x, y, C, gamma)


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
    x = np.load(test_x_file)
    logmsg('Testing the model ...')
    prob = test(model, x)
    logmsg('Saving predictions ...')
    pred = format_pred(prob, data)
    pred.to_csv(pred_file, index=False)


if __name__ == '__main__':
    C = 1
    gamma = 'auto'
    model = do_train(TRAIN_X_FILE, TRAIN_Y_FILE, C, gamma)
    do_test(model, TEST_FILE, TEST_X_FILE, PRED_FILE)
