import matplotlib.pylab as plt
import numpy as np
from numpy import apply_along_axis as apply
from scipy.signal import fftconvolve
import sorting_with_python as swp
from load_data import load_data

raw_data, data_len = load_data()
tt = np.arange(0,data_len)/1.5e4
data = list(map(lambda x: (x-np.median(x))/swp.mad(x), raw_data))

data_filtered = apply(lambda x:
                      fftconvolve(x,np.array([1,1,1,1,1])/5.,'same'),
                      1,np.array(data))
data_filtered = (data_filtered.transpose() / \
                 apply(swp.mad,1,data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0

def print_stats(title, sp):
    print("Stats for %s" % title)
    print("giving %d spikes" % len(sp))
    print("a mean inter-event interval of %f sampling points" % round(np.mean(np.diff(sp))))
    print("a standard deviation of %f sampling points" % round(np.std(np.diff(sp))))
    print("a smallest inter-event interval of %f sampling points" % np.min(np.diff(sp)))
    print("and a largest of %f sampling points" % np.max(np.diff(sp)))
    print("")

sp0 = swp.peak(data_filtered.sum(0))
sp0E = sp0[sp0 <= data_len/2.]
sp0L = sp0[sp0 > data_len/2.]
print_stats("sp0", sp0)
print_stats("sp0E", sp0E)
print_stats("sp0L", sp0L)

plt.figure(1)
plt.subplot(211)
plt.plot(tt, data[0],color='black')
plt.axhline(y=4,color="blue",linestyle="dashed")
plt.plot(tt, data_filtered[0,],color='red')
plt.xlim([0,0.2])
plt.ylim([-5,10])
plt.xlabel('Time (s)')
plt.subplot(212)
swp.plot_data_list_and_detection(data,tt,sp0)
plt.xlim([0,0.2])
plt.show()
