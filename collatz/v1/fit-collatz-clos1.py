from time import process_time

import matplotlib.pyplot as plt
from fitter import Fitter
from scipy.stats import kstest

from collatz.v1.misc import get_fullpath, get_data, logmsg

CLOS_FILE = get_fullpath('data/collatz-clos1-824559.csv')


def get_miss(stop_file):
    data = get_data(stop_file, cols=['miss'])
    return data['miss']


def fit_exp(miss):
    dist = ['expon']
    f = Fitter(miss, distributions=dist, timeout=600)
    f.fit()
    params = f.fitted_param['expon']
    logmsg('fitted params exp = %s', str(params))
    f.summary()
    plt.show()
    return params


def stat_test(x, params):
    start = process_time()
    _, pvalue = kstest(rvs=x, cdf='expon', args=params, mode='asymp')
    end = process_time()
    logmsg('Kolmogorov-Smirnov p-value = %f (%f secs)', pvalue, end - start)


# main
miss = get_miss(CLOS_FILE)
params = fit_exp(miss)
stat_test(miss, params)
