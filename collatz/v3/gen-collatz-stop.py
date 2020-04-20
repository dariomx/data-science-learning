# computes collatz stop function up to certain limit

import pandas as pd

from collatz.v3.misc import get_fullpath, logmsg

MAX_N = 10000000

STOP_FILE = get_fullpath('data/collatz-stop-%d.csv' % MAX_N)
COLS = ['n', 'stop', 'miss']


# count only misses up to first hit (but count the hit too!)
def collatz_stop(n, cache):
    orig_n = n
    stop, miss = 0, 0
    while n != 1:
        miss += 1
        if n in cache:
            stop += cache[n][1]
            break
        else:
            stop += 1
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    rec = orig_n, stop, miss
    cache[orig_n] = rec
    return rec


def gen_stop_seq(max_n, progress=1000000):
    seq = []
    cache = dict()
    for n in range(1, max_n + 1):
        rec = collatz_stop(n, cache)
        if n % progress == 0:
            logmsg('Generated %d stop times so far' % n)
        seq.append(rec)
    return pd.DataFrame(seq, columns=COLS)


# main
seq = gen_stop_seq(MAX_N)
seq.to_csv(STOP_FILE, index=False)
