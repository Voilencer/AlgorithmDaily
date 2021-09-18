import numpy as np
"""
齐次方程 Ax=0的解
A: m x n矩阵, m是方程数， n是未知数的个数

一定有零解
r(A | 0) = r(A)
r(A) = n，约束较强
    方针，  只有唯一的零解
    非方针，m > n 没有真正的非零解析解，因为约束太多，可以求最小二乘法，一般使用SVD
r(A) < n, 约束不够
        有非零解，不唯一

argmin(||Ax||^2)

（1） 特征值和特征向量，最小特征值对应的特征向量
（2）SVD分解
"""

"""
np.linalg.svd(a, full_matrices=True, compute_uv=True
* 奇异值分解
输入
    a —— M x N 矩阵
    full_matrices—— u和vh的输出形状
    compute_uv_s——是否计算u和vh
返回
    u 
    s 
    vh
"""


def solve1(A):
    # A SVD分解，最小特征值对应的特征向量, V最后一列,Vt最后一行
    A = np.matrix(A)
    Q, S, Vt = np.linalg.svd(A)
    return Vt[-1, :]


def solve(A):
    # AtA最小特征值对应特征向量
    A = np.mat(A)
    U = A.T * A
    lamda, hU = np.linalg.eig(U)
    return hU[:, -1]



if __name__ == "__main__":
    A = np.array([[4,5,6], [1,2,3]])
    res1 = solve1(A)

    res2 = solve(A)

    tmp1 = np.matrix(A) * np.matrix(res1).T
    print(tmp1)

    tmp2 = np.matrix(A) * np.matrix(res2)
    print(tmp2)
