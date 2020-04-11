from datetime import datetime as dt

CATLAB = ['toxic',
          'severe_toxic',
          'obscene',
          'threat',
          'insult',
          'identity_hate']
ID = 'id'
COLS = [ID] + CATLAB

NONTOXIC = 'non_toxic'
EXT_CATLAB = CATLAB + [NONTOXIC]
CATIDX = {c: i for (i, c) in enumerate(EXT_CATLAB)}

WID = 'wid'
WORD = 'word'
FREQ = 'freq'
VOCAB_COLS = [WID, WORD, FREQ]

COMMENT = 'comment_text'


def get_row_words(row):
    return set(row[COMMENT].split(' '))


def logmsg(fmt, *args):
    ts = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    print(ts + ': ' + (fmt % args))
