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


def get_row_words(row):
    try:
        return set(row[COMMENT].split())
    except ValueError:
        raise
