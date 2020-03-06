from collections import defaultdict

import nltk
import pandas as pd

from common import *

TRAIN_DATA = '../data/attempt2/pre-train.csv'
TEST_DATA = '../data/attempt2/pre-test.csv'
PRED_DATA = '../data/attempt2/pred-2a.csv'
MAX_VOCAB = 2000

def get_row_words(row):
    return set(row[COMMENT].split())


def get_vocab(data):
    vocab = defaultdict(lambda: 0)
    toxic_data = data[data[CATLAB].sum(axis=1) > 0]
    for _, row in toxic_data.iterrows():
        for w in get_row_words(row):
            vocab[w] += 1
    vocab = sorted(vocab.items(), key=lambda t: -t[1])
    return [w for (w, _) in vocab[:MAX_VOCAB]]


def get_row_feat(row, vocab):
    row_feat = {}
    row_words = get_row_words(row)
    for w in vocab:
        row_feat[w] = w in row_words
    return row_feat


def get_features(data, vocab):
    features = []
    for _, row in data.iterrows():
        for cat in CATLAB:
            row_feat = get_row_feat(row, vocab)
            if row[cat] == 1:
                features.append((row_feat, cat))
    return features


def train(data, vocab):
    feats = get_features(data, vocab)
    return nltk.NaiveBayesClassifier.train(feats)


def predict(data, model, vocab):
    pred = []
    for _, in_row in data.iterrows():
        id = in_row[ID]
        row_feat = get_row_feat(in_row, vocab)
        dist = model.prob_classify(row_feat)
        out_row = {ID: id}
        for cat in CATLAB:
            out_row[cat] = dist.prob(cat)
        pred.append(out_row)
    return pd.DataFrame(pred, columns=COLS)


if __name__ == '__main__':
    print('Reading data ...')
    train_data = pd.read_csv(TRAIN_DATA, dtype={COMMENT: str})
    test_data = pd.read_csv(TEST_DATA, dtype={COMMENT: str})
    print('Building vocabulary ...')
    vocab = get_vocab(train_data)
    print('Training the model ...')
    model = train(train_data, vocab)
    print('Testing the model ...')
    pred_data = predict(test_data, model, vocab)
    print('Saving predictions ...')
    pred_data.to_csv(PRED_DATA, index=False)
