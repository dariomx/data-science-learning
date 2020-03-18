import pandas as pd

from common import *

VOCAB_DATA = '../data/attempt4/vocab.txt'
TRAIN_DATA = '../data/attempt4/pre-train.csv'
TEST_DATA = '../data/attempt4/pre-test.csv'
PRED_DATA = '../data/attempt4/pred-4d.csv'

def get_vocab():
    with open(VOCAB_DATA) as f:
        vocab = [w.strip() for w in f.readlines()]
        return set(vocab)

def get_row_feat(row, vocab):
    row_feat = {}
    row_words = get_row_ngrams(row)
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
    logmsg('Reading vocab ...')
    vocab = get_vocab()
    logmsg('Reading data ...')
    train_data = pd.read_csv(TRAIN_DATA, dtype={COMMENT: str})
    test_data = pd.read_csv(TEST_DATA, dtype={COMMENT: str})
    logmsg('Training the model ...')
    model = train(train_data, vocab)
    logmsg('Testing the model ...')
    pred_data = predict(test_data, model, vocab)
    logmsg('Saving predictions ...')
    pred_data.to_csv(PRED_DATA, index=False)
