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

# not used, and gives error
#noiseE = swp.mk_noise(sp0E,np.array(data),14,30,safety_factor=2.5,size=2000)

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

evt_idx = range(180)
evtsE_good_mean = np.mean(evtsE[goodEvts,:],0)
plt.figure(1)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(evt_idx,evtsE_good_mean, 'black',evt_idx,
             evtsE_good_mean + 5 * u[:,i],
             'red',evt_idx,evtsE_good_mean - 5 * u[:,i], 'blue')
    plt.title('PC' + str(i) + ': ' + str(round(s[i]/sum(s)*100)) +'%')

plt.figure(2)
for i in range(4,8):
    plt.subplot(2,2,i-3)
    plt.plot(evt_idx,evtsE_good_mean, 'black',
             evt_idx,evtsE_good_mean + 5 * u[:,i], 'red',
             evt_idx,evtsE_good_mean - 5 * u[:,i], 'blue')
    plt.title('PC' + str(i) + ': ' + str(round(s[i]/sum(s)*100)) +'%')


# can't calculate due error in noiseE
#noiseVar = sum(np.diag(np.cov(noiseE.T)))
#evtsVar = sum(s)
#[(i,sum(s[:i])+noiseVar-evtsVar) for i in range(15)]

evtsE_good_P0_to_P3 = np.dot(evtsE[goodEvts, :], u[:, 0:4])
df = pd.DataFrame(evtsE_good_P0_to_P3)
scatter_matrix(df, alpha=0.2, s=4, c='k', figsize=(6, 6),
               diagonal='kde', marker=".")

f = open('data/evtsE.csv','w')
w = csv.writer(f)
w.writerows(np.dot(evtsE[goodEvts,:],u[:,:8]))
f.close()

plt.show()

