import nltk
from datetime import datetime as dt

CATLAB = ['toxic',
          'severe_toxic',
          'obscene',
          'threat',
          'insult',
          'identity_hate']
NONTOXIC = 'non_toxic'
EXT_CATLAB = CATLAB + [NONTOXIC]
CATIDX = {c: i for (i, c) in enumerate(EXT_CATLAB)}
COMMENT = 'comment_text'
ID = 'id'
COLS = [ID] + CATLAB
CAT = 'category'
WORD = 'word'
COUNT = 'count'
TOTAL = 'TOTAL'
TOTAL_V = 'TOTAL_V'
VID = 'vid'
NGRAM = 'n-gram'
VOCAB_COLS = [VID, NGRAM, 'df', 'idf']
MAX_VOCAB = {1: 2000, 2: 1000, 3: 500, 4: 250, 5: 125}
BAD_VOCAB = 0
GOOD_VOCAB = 1


def get_row_ngrams(row):
    try:
        words = row[COMMENT].split()
    except AttributeError:
        return set()
    ngrs = []
    for n in MAX_VOCAB:
        ngrs += nltk.ngrams(words, n)
    ngrs = [' '.join(w) for w in ngrs]
    return set(ngrs)


def logmsg(fmt, *args):
    ts = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    print(ts + ': ' + (fmt % args))
