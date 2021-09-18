import numpy as np

"""
A x = lamda x
* A是方阵

* 
W， V = np.linalg.eig()
计算一个方阵的特征值和右特征向量
特征值不会排序
V是归一化后的特征向量 
W[i] 和 V[:,i]对应    A * V[:,i] = W[i] * V[:,i]
"""


def get_eig(A):
    assert A.shape[0] == A.shape[1]
    det = np.linalg.det(A)
    assert det != 0
    W, Vt = np.linalg.eig(A)
    print('特征值：', W)
    print('特征向量：', Vt)

    # 验证Ax=lamdax
    res1 = np.dot(A, Vt[:,0])
    res2 = W[0] * Vt[:,0]

    # 归一化的特征向量
    res = np.dot(Vt[:,0], Vt[:,0].T)
    print(res)







if __name__ == "__main__":
    A = np.array([[2, 3], [4, 5]])
    get_eig(A)
