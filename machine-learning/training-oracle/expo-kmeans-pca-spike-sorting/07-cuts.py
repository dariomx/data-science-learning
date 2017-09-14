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

sp0 = swp.peak(data_filtered.sum(0))
sp0E = sp0[sp0 <= data_len/2.]
sp0L = sp0[sp0 > data_len/2.]

evtsE = swp.mk_events(sp0E,np.array(data),49,50)
evtsE_median=apply(np.median,0,evtsE)
evtsE_mad=apply(swp.mad,0,evtsE)

plt.figure(1)
plt.subplot(221)
plt.plot(evtsE_median, color='red', lw=2)
plt.axhline(y=0, color='black')
for i in np.arange(0,400,100):
    plt.axvline(x=i, color='black', lw=2)
for i in np.arange(0,400,10):
    plt.axvline(x=i, color='grey')
plt.plot(evtsE_median, color='red', lw=2)
plt.plot(evtsE_mad, color='blue', lw=2)

evtsE = swp.mk_events(sp0E,np.array(data),14,30)
plt.subplot(222)
swp.plot_events(evtsE,200)

# not used, and gives error
# noiseE = swp.mk_noise(sp0E,np.array(data),14,30,safety_factor=2.5,size=2000)

def good_evts_fct(samp, thr=3):
    samp_med = apply(np.median,0,samp)
    samp_mad = apply(swp.mad,0,samp)
    above = samp_med > 0
    samp_r = samp.copy()
    for i in range(samp.shape[0]): samp_r[i,above] = 0
    samp_med[above] = 0
    res = apply(lambda x:
                np.all(abs((x-samp_med)/samp_mad) < thr),
                1,samp_r)
    return res

goodEvts = good_evts_fct(evtsE,8)
plt.subplot(223)
swp.plot_events(evtsE[goodEvts,:][:200,:])

plt.subplot(224)
swp.plot_events(evtsE[~goodEvts,:],
                show_median=False,
                show_mad=False)

plt.show()