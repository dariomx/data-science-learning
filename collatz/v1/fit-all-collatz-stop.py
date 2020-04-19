import sys
import warnings

import matplotlib.pyplot as plt
from fitter import Fitter

from collatz.v1.misc import get_fullpath, get_data, logmsg

CLOS_FILE = get_fullpath('data/collatz-stop.csv')


def get_col(stop_file, col):
    data = get_data(stop_file, cols=[col])
    return data[col]


def fit_all(miss):
    # dist = get_distributions()
    # dist = ['genpareto', 'betaprima', 'lomax', 'f', 'ncf']
    # dist = ['genpareto', 'lomax', 'f', 'ncf']
    # dist = ['genpareto', 'lomax']
    # dist = ['expon', 'lomax']
    dist = [ 'expon']
    f = Fitter(miss, timeout=600, distributions=dist)
    f.fit()
    print(f.df_errors.sort_values('sumsquare_error'))
    logmsg('best fit = %s', str(f.get_best()))
    f.summary()
    plt.show()


# main
if not sys.warnoptions:
    warnings.simplefilter("ignore")
x = get_col(CLOS_FILE, 'miss')
#x = x[:1001] # 703 seems the magical number to fit exp
#x = x[:10000+1]
#x = x[:100000+1]
x = x[:1000000+1]
fit_all(x)
