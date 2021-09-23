"""线性方程组
AX = B
A： m x n
可以通过增广矩阵来判断
r(A) < r(A | B) 无解
r(A) = r(A | B) = n  唯一解
r(A) = r(A | B) < n  无穷解
r(A) > r(A | B)  不可能

A =
1 2 2 2
2 4 6 8
3 6 8 10
x = [x1, x2, x3, x4]
b = [b1, b2, b3]
"""


def solve_1():
    # (1) 增广矩阵
    # (2)  初等行变换
    # (3) 分析秩
    pass



import numpy as np
a = np.array([[2, -2, -4], [-1, 3, 4], [1, -2, -3]])
s, v, d = np.linalg.svd(a)
print(v)
print(d)
np.compress(v < 1e-10, d, axis=0)
res = np.array([[ 0.57735027, -0.57735027,  0.57735027]]) # 通解
print(res)






if __name__ == "__main__":
    pass