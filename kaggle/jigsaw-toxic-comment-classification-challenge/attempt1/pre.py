# Preprocessing here means converting raw comments into a list of
# normalized words
#
# Disable stemming/lemmarization as they seem not really needed
# (slightly smaller score with them).

import string

import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

IN_OUT = {
    '../data/train.csv': '../data/attempt1/pre-train.csv',
    '../data/test.csv': '../data/attempt1/pre-test.csv'
}

from common import COMMENT

PUNCT_TABLE = str.maketrans('', '', string.punctuation)
STOP_WORDS = set(stopwords.words('english'))


def norm_word(word):
    word = word.encode('ascii', errors='ignore').decode()
    word = word.translate(PUNCT_TABLE)
    word = word.lower()
    return word


def good_word(word):
    return word.isalpha() and word not in STOP_WORDS


def norm_text(text):
    words = [norm_word(w) for s in sent_tokenize(text) \
             for w in word_tokenize(s)]
    words = [w for w in words if good_word(w)]
    return ' '.join(words)


def preproc(in_file, out_file):
    data = pd.read_csv(in_file)
    data[COMMENT] = data[COMMENT].apply(norm_text)
    data.to_csv(out_file, index=False)


if __name__ == '__main__':
    for in_file, out_file in IN_OUT.items():
        print('pre-processing %s ...' % in_file)
        preproc(in_file, out_file)
