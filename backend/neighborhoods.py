import numpy as np


# long range neighborhood
def make_anisotropic_nhood(xy_ranges=[1, 3, 9, 27]):
    long_range = len(xy_ranges)
    nhood = []
    for i in range(long_range):
        nhood_z = [- (i+1), 0, 0]
        nhood.append(nhood_z)
        range_xy = - xy_ranges[i]
        nhood_y = [0, range_xy, 0]
        nhood.append(nhood_y)
        nhood_x = [0, 0, range_xy]
        nhood.append(nhood_x)
    return np.array(nhood, dtype='int32')


def default_anisotropic_lr():
    return make_anisotropic_nhood()
