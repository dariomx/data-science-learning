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
VOCAB_COLS = ['word', 'df', 'idf', 'dfxidf']
MAX_NGRAMS = 1
MAX_VOCAB = 100000

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


def logmsg(msg):
    ts = dt.now().strftime('%b %d %Y %H:%M:%S')
    print(ts + ': ' + msg)
