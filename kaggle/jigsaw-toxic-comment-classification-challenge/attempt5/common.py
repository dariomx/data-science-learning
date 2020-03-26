from datetime import datetime as dt

import nltk

CATLAB = ['toxic',
          'severe_toxic',
          'obscene',
          'threat',
          'insult',
          'identity_hate']
COMMENT = 'comment_text'
ID = 'id'
COLS = [ID] + CATLAB
CAT = 'category'
WORD = 'word'
COUNT = 'count'
TOTAL = 'TOTAL'
TOTAL_V = 'TOTAL_V'
MAX_VOCAB = {1: 250, 2: 200, 3: 150, 4: 100, 5: 50}


def get_row_ngrams(row):
    try:
        words = row[COMMENT].split()
    except AttributeError:
        return set()
    ngrs = []
    for k in MAX_VOCAB:
        ngrs += nltk.ngrams(words, k)
    ngrs = [' '.join(w) for w in ngrs]
    return set(ngrs)


def logmsg(msg):
    ts = dt.now().strftime('%b %d %Y %H:%M:%S')
    print(ts + ': ' + msg)
