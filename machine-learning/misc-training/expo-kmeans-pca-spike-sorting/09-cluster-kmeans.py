import matplotlib.pylab as plt
import numpy as np
from numpy import apply_along_axis as apply
from scipy.signal import fftconvolve
import sorting_with_python as swp
from load_data import load_data
from numpy.linalg import svd
from pandas.plotting import scatter_matrix
import pandas as pd
import csv
from sklearn.cluster import KMeans

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

evtsE = swp.mk_events(sp0E,np.array(data),14,30)

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

varcovmat = np.cov(evtsE[goodEvts,:].T)
u, s, v = svd(varcovmat)

km10 = KMeans(n_clusters=10, init='k-means++', n_init=100, max_iter=100)
#km10.fit(np.dot(evtsE[goodEvts,:],u[:,0:3]))
c10 = km10.fit_predict(np.dot(evtsE[goodEvts,:],u[:,0:3]))

cluster_median = list([(i,
                        np.apply_along_axis(np.median,0,
                                            evtsE[goodEvts,:][c10 == i,:]))
                                            for i in range(10)
                                            if sum(c10 == i) > 0])
cluster_size = list([np.sum(np.abs(x[1])) for x in cluster_median])
new_order = list(reversed(np.argsort(cluster_size)))
new_order_reverse = sorted(range(len(new_order)), key=new_order.__getitem__)
c10b = [new_order_reverse[i] for i in c10]

f = open('data/evtsEsorted.csv','w')
w = csv.writer(f)
w.writerows(np.concatenate((np.dot(evtsE[goodEvts,:],u[:,:8]),
                            np.array([c10b]).T),
                            axis=1))
f.close()

plt.figure(1)
plt.subplot(511)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 0,:])
plt.ylim([-15,20])
plt.subplot(512)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 1,:])
plt.ylim([-15,20])
plt.subplot(513)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 2,:])
plt.ylim([-15,20])
plt.subplot(514)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 3,:])
plt.ylim([-15,20])
plt.subplot(515)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 4,:])
plt.ylim([-15,20])

plt.figure(2)
plt.subplot(511)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 5,:])
plt.ylim([-10,10])
plt.subplot(512)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 6,:])
plt.ylim([-10,10])
plt.subplot(513)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 7,:])
plt.ylim([-10,10])
plt.subplot(514)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 8,:])
plt.ylim([-10,10])
plt.subplot(515)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 9,:])
plt.ylim([-10,10])

plt.show()
