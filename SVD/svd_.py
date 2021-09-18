import numpy as np


"""SVD分解
np.linalg.svd
A: mxn  U: mxm  V: nxn


"""


def cal_svd(A):
    u, s, vh = np.linalg.svd(A)

    print(u)
    print(s)
    print(vh)




if __name__ == "__main__":
    A = np.matrix([[2,3,4], [5,6,7]])
    cal_svd(A)