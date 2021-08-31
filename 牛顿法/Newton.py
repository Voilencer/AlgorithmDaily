import numpy as np
import random
import matplotlib.pyplot as plt


"""牛顿法
求解非线性方程组
    一元非线性方程
非线性目标函数最小值
"""


def cal_equation1():
    # 非线性一元方程
    # f(x) = x^2 - 7x +12
    def func(x):
        return x**2 - 7 * x +12
    def derivation(x):
        return 2 * x - 7
    x0 = 5
    while abs(func(x0)) > 1e-5:
        x = x0 - func(x0) / derivation(x0)
        x0 = x
    print(x0)


def cal_equation2():
    # x^2 + y^2 +z^2 - 14
    # 多元方程组
    def func(x):
        return np.matrix([
            [x[0][0]**2 + 2 * x[1][0] + 3 * x[2][0] - 14],
            [x[0][0] + x[1][0]**2 + 3 * x[2][0] - 14],
            [x[0][0] + 2 * x[1][0] + x[2][0]**2 - 14]
        ])

    def jacobi(x):
        return np.matrix([
            [2*x[0][0], 2, 3],
            [1, 2*x[1][0], 3],
            [1, 2, 2 * x[2][0]]
        ])
    x0 = np.ones((3, 1))
    while not all(abs(func(x0))) < 1e-5:
        x = x0 - jacobi(x0).I * func(x0)
        x0 = x.A
    print(x0)


def cal_optimation():
    # f(x,y) = 2*x^2 - 2*x*y + y^2 -2x
    def func(x,y):
        return 2*x**2 - 2*x*y + y**2 - 2*x

    def gradient(x):
        return np.matrix([
            [4*x[0][0] - 2*x[1][0]-2],
            [-2*x[0][0] + 2*x[1][0]]
        ])

    def hessian(x):
        return np.matrix([
            [4, -2],
            [-2, 2]
        ])

    x0 = 10 * np.ones((2,1))
    while not all (abs(gradient(x0)) < 1e-5):
        x = x0 - hessian(x0).I * gradient(x0)
        x0 = x.A
    print("极值为:", x0)






if __name__ == "__main__":
    cal_optimation()