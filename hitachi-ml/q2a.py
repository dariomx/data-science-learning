from os.path import join

import numpy as np
import pandas as pd
from scipy.spatial.ckdtree import cKDTree
from sklearn.neighbors import NearestNeighbors

DATA_DIR = 'data'
PRE_DATA = join(DATA_DIR, 'pre.csv')
POST_DATA = join(DATA_DIR, 'post.csv')
DATAGEN_DIR = 'datagen'
OUT_DATA = join(DATAGEN_DIR, 'q2a.csv')


def get_coords(data_file):
    coords = pd.read_csv(data_file, usecols=['XREL', 'YREL'])
    coords.rename(columns={'XREL': 'x', 'YREL': 'y'}, inplace=True)
    return coords


# counts number of (pre) nearest-neighbors in given radius for each post point
def count_nn_radius(post_data, pre_data, radius):
    kdt = cKDTree(pre_data)
    cnt = []
    for _, row in post_data.iterrows():
        cnt.append(len(kdt.query_ball_point(row.values, radius)))
    return np.array(cnt)


# gets the (pre) nearest-neighbors for each given post point
def get_nn(post_data, pre_data, k):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(pre_data)
    _, nn_ix = nn.kneighbors(post_data)
    return pd.DataFrame(pre_data.iloc[nn_ix.flatten(), :])

def combine_post_pre(post_data, pre_data):
    ans = pd.DataFrame(post_data)
    post_data /= 1000  # handle all calculations in microns
    ans['nn500'] = count_nn_radius(post_data, pre_data, 500)
    nn = get_nn(post_data, pre_data, k=1)
    ans['x2'] = nn['x'].values
    ans['y2'] = nn['y'].values
    return ans


if __name__ == '__main__':
    post_data = get_coords(POST_DATA)
    pre_data = get_coords(PRE_DATA)
    ans = combine_post_pre(post_data, pre_data)
    print(ans)
    ans.to_csv(OUT_DATA, index=False)
