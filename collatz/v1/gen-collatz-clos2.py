# computes collatz closures with method 1

import pandas as pd
from sortedcontainers import SortedSet

from collatz.v1.misc import get_fullpath, logmsg

START_N = 100

CLOS_FILE = get_fullpath('data/collatz-clos2-%d.csv' % START_N)
COLS = ['n', 'stop', 'miss']


def collatz_stop(n, cache, pend):
    orig_n = n
    stop = 0
    miss = 0
    while n != 1:
        if n in cache:
            stop = stop + cache[n][0]
            break
        else:
            miss += 1
            pend.add(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        stop += 1
    cache[orig_n] = stop, miss


def gen_clos_seq(start_n, progress=10):
    cache = dict()
    pend = SortedSet(range(1, start_n+1))
    while pend:
        n = pend.pop()
        collatz_stop(n, cache, pend)
        pend -= cache.keys()
        if len(cache) % progress == 0:
            logmsg('Cache size %d, misses %d', len(cache), len(pend))
    seq = [(n, stop, miss) for (n, (stop, miss)) in cache.items()]
    return pd.DataFrame(seq, columns=COLS)


# main
seq = gen_clos_seq(START_N)
seq.to_csv(CLOS_FILE, index=False)
