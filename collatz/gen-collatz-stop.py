# computes collatz stop function up to certain limit

import pandas as pd

from collatz.misc import get_fullpath, logmsg

MAX_N = 100000000

STOP_FILE = get_fullpath('data/collatz-stop.csv')
COLS = ['n', 'stop', 'miss']
cache = dict()


def collatz_stop(n):
    orig_n = n
    stop = 0
    miss = 0
    while n != 1:
        if n in cache:
            stop = stop + cache[n]
            break
        else:
            miss += 1
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        stop += 1
    cache[orig_n] = stop
    return stop, miss


def gen_stop_seq(max_n, progress=1000000):
    seq = []
    for n in range(1, max_n + 1):
        stop, miss = collatz_stop(n)
        if n % progress == 0:
            logmsg('Generated %d stop times so far' % n)
        seq.append((n, stop, miss))
    return pd.DataFrame(seq, columns=COLS)


# main
stop = gen_stop_seq(MAX_N)
stop.to_csv(STOP_FILE, index=False)
