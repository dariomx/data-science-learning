"""
Using the counters we compute the probabilities of each
category given the comment:

p(cat | comment)

per definition of conditional probability:

= p(comment | cat) p(cat) / p (comment)

let us consider comment a sequence of words (just need that on numerator):

= p(w1...wk | cat) p (cat) / p (comment)

per Naïve Bayes (along with its assumptions):

= p(w1 | cat) * ... * p (wk | cat) * p (cat) / p (comment)

Given that p(comment) = 1/n, with n = #comments

= p(w1 | cat) * ... * p (wk | cat) * p (cat) * n

but since p(cat) = cnt[cat] / n

= p(w1 | cat) * ... * p (wk | cat) * cnt[cat]

and to avoid multiplying small quantities:

= exp(log p(w1 | cat) + ... + log p (wk | cat) + log cnt[cat])

where

p(wi | cat) = cnt[cat][wi] / sum(cnt[cat][wj]), for wj in V

where V = vocabulary, aka, set of all words

though we end up applying the correction for unknown words:

p(wi | cat) = cnt[cat][wi] + 1 / sum(cnt[cat][wj]) + len(V)

"""
from collections import defaultdict

import pandas as pd
from math import log, exp

from common import *

CNT_DATA = '../data/attempt1/cnt.csv'
TEST_DATA = '../data/attempt1/pre-test.csv'
PRED_DATA = '../data/attempt1/pred.csv'


def parse_counters(cnt_data):
    cnt = defaultdict(lambda: defaultdict(lambda: 0))
    V = set()
    for _, row in pd.read_csv(cnt_data).iterrows():
        cat = row[CAT]
        word = row[WORD]
        count = row[COUNT]
        cnt[cat][word] = count
        V.add(word)
    return cnt, V


def calc_prob(cat, comment, cnt, V):
    if len(comment) == 0:
        return 0
    if TOTAL_V not in cnt[cat]:
        for word in V:
            cnt[cat][TOTAL_V] += cnt[cat][word]
    prob = log(cnt[cat][TOTAL])
    for word in comment.split():
        prob += log((cnt[cat][word] + 1) / (cnt[cat][TOTAL_V] + len(V)))
    if prob > 0:  # not enough known words?
        prob = 0
    else:
        prob = exp(prob)
    return prob

def predict(data, cnt, V):
    pred = []
    for _, in_row in data.iterrows():
        id = in_row[ID]
        comment = str(in_row[COMMENT])
        out_row = {ID: id}
        for cat in CATLAB:
            prob = calc_prob(cat, comment, cnt, V)
            # print('p(%s | %s) = %f' % (cat, comment, prob))
            out_row[cat] = prob
        pred.append(out_row)
    return pd.DataFrame(pred, columns=COLS)


if __name__ == '__main__':
    cnt, V = parse_counters(CNT_DATA)
    data = pd.read_csv(TEST_DATA)
    pred = predict(data, cnt, V)
    pred.to_csv(PRED_DATA, index=False)
