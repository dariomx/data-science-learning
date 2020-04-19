import matplotlib.pyplot as plt
from fitter import Fitter

from collatz.v1.misc import get_fullpath, get_data

STOP_FILE = get_fullpath('data/collatz-stop.csv')


def get_miss(stop_file):
    data = get_data(stop_file, cols=['miss'])
    return data['miss']


def fit_exp(miss, dist):
    dist = [dist]
    f = Fitter(miss, distributions=dist, timeout=600, verbose=True)
    f.fit()
    return f.df_errors['sumsquare_error'][dist]


def plot_fit(miss):
    step = 100000
    x = list(range(step, 1000 * step + 1, step))
    y = []
    for ss in x:
        y.append(fit_exp(miss[:(ss + 1)]))
    plt.plot(x, y)
    plt.show()


# main
miss = get_miss(STOP_FILE)
fit_exp(miss, 'pareto')
