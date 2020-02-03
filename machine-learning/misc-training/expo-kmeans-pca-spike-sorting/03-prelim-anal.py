import numpy as np
from load_data import load_data
from scipy.stats.mstats import mquantiles

data, data_len = load_data()

print("3.0 Data size")
print("data_len = %d" % data_len)
print("")

print("3.1 Five number summary")
np.set_printoptions(precision=3)
for row in [mquantiles(x,prob=[0,0.25,0.5,0.75,1]) for x in data]:
    print(row)
print("")

print("3.2 Were the data normalized?")
for x in data:
    print(np.std(x))
print("")

print("3.3 Discretization step amplitude")
for x in data:
    print(np.min(np.diff(np.sort(np.unique(x)))))
print("")

