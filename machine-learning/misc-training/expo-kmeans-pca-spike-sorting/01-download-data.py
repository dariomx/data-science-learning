from urllib import urlretrieve
from itertools import izip

data_dir = "./data"
data_names = ['Locust_' + str(i) + '.dat.gz' for i in xrange(1,5)]
data_src = ['http://xtof.disque.math.cnrs.fr/data/' + n
            for n in data_names]

for src,dst in izip(data_src, data_names):
    local_dst = data_dir + "/" + dst
    print("downloading %s -> %s" % (src, local_dst))
    urlretrieve(src,local_dst)
