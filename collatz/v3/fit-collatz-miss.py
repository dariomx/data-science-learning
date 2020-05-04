# computes collatz stop function up to certain limit
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import geom, chisquare

from collatz.v3.misc import get_fullpath, get_data
from collatz.v3.misc import logmsg

MAX_N = 10000000

STOP_FILE = get_fullpath('data/collatz-stop-%d.csv' % MAX_N)
COLS = ['n', 'stop', 'miss']



def get_odd_miss(stop_file):
    miss = get_data(stop_file)['miss']
    odd_ix = list(range(0, len(miss), 2))
    return miss.iloc[odd_ix]

# MLE formula for estimating p (fst variant, according to wikipedia)
# TODO: why loc=0, should not be 1? but plot looks better with 0
def fit_geom(miss):
    n = len(miss)
    p = n / sum(miss)
    loc = 0
    params = p, loc
    logmsg('mle params = %s', str(params))
    return params


def plot_geom(miss, p, loc):
    plt.hist(miss, density=True, bins=1000)
    x = list(range(1, max(miss) + 1))
    y = geom.pmf(x, p=p, loc=loc)
    plt.plot(x, y, lw=2)
    plt.show()


def stat_test(miss, params):
    cnt = sorted(Counter(miss).items(), key=lambda t: t[0])
    x, freq = zip(*cnt)
    x, freq = np.array(x), np.array(freq)
    freq = freq / sum(freq)
    # TODO: freq key to match https://oeis.org/A186323?
    print(','.join([str(k) for k, _ in cnt]))
    print(','.join(map(str, freq)))
    pmf = geom.pmf(x, *params)
    _, pvalue = chisquare(freq, pmf)
    logmsg('p-value = %f', pvalue)
    return pmf[1], pvalue


def plot_err(miss, step=100000):
    x = range(step, len(miss) + 1, step)
    y = []
    for size in x:
        s_miss = miss[:(size + 1)]
        params = fit_geom(s_miss)
        err, _ = stat_test(s_miss, params)
        y.append(err)
        logmsg('error %f for size %d ...', err, size)
    plt.plot(x, y)
    plt.xlabel('sample size')
    plt.title('geom fit error')
    plt.show()


# main
miss = get_odd_miss(STOP_FILE)
#plot_err(miss)
params = fit_geom(miss)
plot_geom(miss, *params)
stat_test(miss, params)