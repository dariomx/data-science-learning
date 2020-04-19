# computes collatz stop function up to certain limit

import matplotlib.pyplot as plt
import pandas as pd

from collatz.v1.misc import get_fullpath, logmsg

MAX_N = 1000000

STOP_FILE = get_fullpath('data/collatz-stop-%d.csv' % MAX_N)
COLS = ['n', 'stop', 'hits', 'miss']


# count only misses up to first hit
def collatz_stop(n, cache):
    orig_n = n
    stop, miss = 0, 0
    while n != 1:
        if n in cache:
            stop += cache[n][1]
            break
        else:
            miss += 1
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        stop += 1
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


def plot_probs(seq, step=10000, progress=1000000):
    p_hits = []
    for size in range(step, len(seq) + 1, step):
        s_seq = seq[:size]
        ph = s_seq['hits'] / s_seq['stop']
        p_hits.append((size, ph.mean(), ph.median()))
        if size % progress == 0:
            logmsg('Calculated probs up to size %d ...' % size)
    p_hits = pd.DataFrame(p_hits, columns=['size', 'mean', 'median'])
    p_hits.plot('size', ['mean', 'median'])
    plt.show()

# main
seq = gen_stop_seq(MAX_N)
plot_probs(seq)
# seq.to_csv(STOP_FILE, index=False)
