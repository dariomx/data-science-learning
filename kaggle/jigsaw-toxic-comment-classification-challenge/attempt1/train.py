"""
Training here means to compute the counters for each (category, word) pair,
as well for each category alone. these counters are the precursors of the
probabilities we need for naïve-bayes, but prefer to avoid storing the
probabilities due both space and rounding errors.

The output will be just a csv file, with category, word, counter columns. The
special word TOTAL will be used to set category totals (no clash with regular
words, as they are expected to be in lower case due pre-processing).

"""
import csv
from collections import defaultdict

import pandas as pd

from common import *

IN_DATA = '../data/attempt1/pre-train.csv'
OUT_DATA = '../data/attempt1/cnt.csv'


def calc_counters(data):
    cnt = defaultdict(lambda: defaultdict(lambda: 0))
    for _, row in data.iterrows():
        comment = row[COMMENT]
        if type(comment) != str or len(comment) == 0:
            continue  # no words left after normalization
        words = comment.split()
        for cat in CATLAB:
            if row[cat] == 0:
                continue
            for word in words:
                cnt[cat][word] += 1
            cnt[cat][TOTAL] += 1
        cnt[TOTAL][TOTAL] += 1
    return cnt


def save_counters(cnt, out_file):
    with open(out_file, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([CAT, WORD, COUNT])
        for cat, word_count in cnt.items():
            for word, count in word_count.items():
                writer.writerow([cat, word, count])


if __name__ == '__main__':
    # TODO: is dtype worth it? runtime reveals that it gets float for empty
    data = pd.read_csv(IN_DATA, dtype={COMMENT: str})
    cnt = calc_counters(data)
    save_counters(cnt, OUT_DATA)
