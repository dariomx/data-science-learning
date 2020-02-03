# Will use SVD to reduce dimension, whose first two axis cover more than
# 99% of the total variability. Not totally sure that normalization is
# required for this particular scenario, not applying it for now.
#
# The resulting data is visualized into a 2D scatter plot, and we can appreciate
# that most of the information lies around a linear relationship; which is
# perhaps easier to understand/analyze further than the multidimensional data
# we had first. There is a sort of outlier on the right-down corner, which
# probably should be analyzed separately.
#

from os.path import join

import pandas as pd
from matplotlib import pyplot as plt
from numpy import transpose
from sklearn.decomposition import TruncatedSVD

from utils import new_fig, save_plot

DATA_DIR = 'datagen'
DATA_FILE = join(DATA_DIR, 'all-toollog.csv')
PLOTS_DIR = 'plots'
PLOT_FILE = join(PLOTS_DIR, 'q1d.png')

# we also discard the serial numbers, as they do not make sense in dimension
# reduction.
def get_data(data_file):
    data = pd.read_csv(data_file)
    good_cols = []
    for col in data.columns:
        col_low = col.lower()
        if 'time' in col_low or 'date' in col_low or 's/n' in col_low or \
                col == 'Fault Number of ten times in the past':
            print('Discarding column %s ' % col)
            continue
        good_cols.append(col)
    data = data.loc[:, data.columns.isin(good_cols)]
    return data

def reduce_dim(data, dim=2):
    svd = TruncatedSVD(n_components=dim)
    print('Reducing dimensionality with SVD')
    svd.fit(data)
    print('Variability explained by reduced '
          'data %f' % svd.explained_variance_ratio_.sum())
    # beware scikit-learn methods usually return matrix transposed
    return pd.DataFrame(transpose(svd.components_))

def plot_data_2d(data, plot_file):
    fig = new_fig('Reduced data')
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    plt.scatter(x, y)
    save_plot(plot_file, fig)
    print('Saved plot in %s' % plot_file)


if __name__ == '__main__':
    data = get_data(DATA_FILE)
    data_2d = reduce_dim(data, dim=2)
    plot_data_2d(data_2d, PLOT_FILE)
