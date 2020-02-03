import numpy as np
from scipy.stats.mstats_basic import mquantiles
import sorting_with_python as swp
from load_data import load_data
import matplotlib.pylab as plt
from scipy.stats import norm

raw_data, data_len = load_data()
tt = np.arange(0,data_len)/1.5e4

print("5.0 Data renormalization with MAD")
data_mad = list(map(swp.mad, raw_data))
print("data_mad = %s" % str(data_mad))
print("")

data = list(map(lambda x: (x-np.median(x))/swp.mad(x), raw_data))

def plot_mad(figno, xmax):
    plt.figure(figno)
    plt.ylim([-5,10])
    plt.subplot(211)
    plt.plot(tt, raw_data[0],color="black")
    plt.xlim([0, xmax])
    plt.ylim([-17,13])
    plt.xlabel('Time (s)')
    plt.subplot(212)
    plt.plot(tt, data[0],color="black")
    plt.xlim([0, xmax])
    plt.ylim([-17,13])
    plt.xlabel('Time (s)')

def plot_qq(figno):
    plt.figure(figno)
    plt.subplot(111)
    dataQ = map(lambda x:
                mquantiles(x, prob=np.arange(0.01, 0.99, 0.001)), data)
    dataQsd = map(lambda x:
                  mquantiles(x / np.std(x), prob=np.arange(0.01, 0.99, 0.001)),
                  data)
    qq = norm.ppf(np.arange(0.01, 0.99, 0.001))
    plt.plot(np.linspace(-3, 3, num=100), np.linspace(-3, 3, num=100),
             color='grey')
    colors = ['black', 'orange', 'blue', 'red']
    for i, y in enumerate(dataQ):
        plt.plt.plot(qq, y, color=colors[i])
    for i, y in enumerate(dataQsd):
        plt.plot(qq, y, color=colors[i], linestyle="dashed")
    plt.xlabel('Normal quantiles')
    plt.ylabel('Empirical quantiles')

plot_mad(1, 20)
plot_mad(2, 0.2)
plot_qq(3)
plt.show()


