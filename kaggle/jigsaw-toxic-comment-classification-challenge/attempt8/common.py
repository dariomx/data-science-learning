from datetime import datetime as dt

import nltk

CATLAB = ['toxic',
          'severe_toxic',
          'obscene',
          'threat',
          'insult',
          'identity_hate']
CATIDX = {c: i for (i, c) in enumerate(CATLAB)}
COMMENT = 'comment_text'
ID = 'id'
COLS = [ID] + CATLAB
CAT = 'category'
WORD = 'word'
COUNT = 'count'
TOTAL = 'TOTAL'
TOTAL_V = 'TOTAL_V'
VOCAB_COLS = ['word', 'df', 'idf']
MAX_NGRAMS = 1
MAX_VOCAB = 2000

def get_row_ngrams(row):
    try:
        words = row[COMMENT].split()
    except AttributeError:
        return set()
    ngrs = []
    for k in range(1, MAX_NGRAMS+1):
        ngrs += nltk.ngrams(words, k)
    ngrs = [' '.join(w) for w in ngrs]
    return set(ngrs)


def logmsg(fmt, *args):
    ts = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    print(ts + ': ' + (fmt % args))
