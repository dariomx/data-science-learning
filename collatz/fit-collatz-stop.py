# TODO: fitting stopping time to a dist seems harder, better see miss seq

import matplotlib.pyplot as plt
from fitter import Fitter
from scipy.stats import normaltest

from collatz.misc import get_fullpath, get_data, logmsg

STOP_FILE = get_fullpath('data/collatz-stop.csv')
stop = get_data(STOP_FILE)['stop']
plt.hist(stop, density=True, bins=100)
plt.show()

_, pvalue = normaltest(stop)
logmsg('Normality test p-value = %.20f', pvalue)

f = Fitter(stop.sample(300000), timeout=60)
f.fit()
f.summary()
