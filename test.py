"""
check the correctness of functions
"""

import numpy as np
from nmp.conversion import *


if __name__ == "__main__":
    r = np.random.normal(np.zeros([10, 3]), 1)
    q = expmap2quat(r)
    rec_r_1 = quat2expmap(q)
    error1 = np.max(np.abs(rec_r_1 - r))

    R = expmap2rotmat(r)
    rec_r_2 = rotmat2expmap(R)
    error2 = np.max(np.abs(rec_r_2 - r))

    print("error1: {}".format(error1))
    print("error2: {}".format(error2))