from time import process_time

import matplotlib.pyplot as plt
from fitter import Fitter
from scipy.stats import anderson

from collatz.misc import get_fullpath, get_data, logmsg

STOP_FILE = get_fullpath('data/collatz-stop.csv')


def get_miss(stop_file):
    data = get_data(stop_file, cols=['miss'])
    return data['miss']


# dist = ['pareto', 'gilbrat', 'pearson3', 'wald', 'beta',
#        'powerlognorm', 'gengamma', 'expon',
#        'exponnorm', 'genexpon', 'exponweib', 'erlang']

# dist = ['pearson3', 'wald', 'exponnorm', 'expon', 'genexpon', 'gilbrat'
#        'gengamma', 'pareto']

# f = Fitter(miss.sample(10000000), distributions=dist, timeout=600)

def fit_exp(miss):
    dist = ['expon']
    f = Fitter(miss, distributions=dist, timeout=600)
    f.fit()
    logmsg('fitted params exp = %s', str(f.fitted_param))
    # f.summary()
    # plt.show()
    return f.fitted_param['expon'][1]


def stat_test(x):
    start = process_time()
    # _, pval = lilliefors(x, dist='exp')
    stat, critval, alpha = anderson(x, dist='expon')
    end = process_time()
    logmsg('sample_size=%d: Anderson: %s %s %s (%f secs)',
           len(x), str(stat), str(critval), str(alpha), end - start)


def plot_pvalue(pvalue):
    plt.plot(range(1, 101), pvalue)
    plt.title('Stat test vs sample size')
    plt.xlabel('sample size (millions)')
    plt.ylabel('p-value')
    plt.show()


# dont pass that much data to statistical test?
# sample_sizes_m = list(range(million, 100 * million + 1, million))
def calc_pvalue(miss):
    pvalue = []
    step = 100
    sample_sizes_m = list(range(step, step * 100 + 1, step))
    for ssm in sample_sizes_m:
        pval = stat_test(miss.sample(ssm))
        pvalue.append(pval)
    return pvalue


def calc_lambda(miss):
    million = 10 ** 6
    sample_sizes_m = list(range(million, 100 * million + 1, million))
    lam = []
    for ssm in sample_sizes_m:
        lam.append(fit_exp(miss[:ssm]))
    return sample_sizes_m, lam


# main
miss = get_miss(STOP_FILE)
ssm, lam = calc_lambda(miss)
plt.plot(ssm, miss)
plt.show()
