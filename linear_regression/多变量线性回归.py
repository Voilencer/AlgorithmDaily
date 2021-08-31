import numpy as np
import random
import matplotlib.pyplot as plt

""" 多元线性回归
模型函数： h(x) = θ0 + θ1 * x1 + θ2 * x2 + ... θn * xn

代价函数 J(θ) = 1/2m sum(（h(i)-y(i)）^2)
    J'(θ) = 1/m Bt * (BX - L)
梯度下降法：   θ = θ0 - learning_rate * J'(θ)
    可以用梯度值大小作为迭代终止条件
"""



def cost_function(theta, X, Y):
    B = np.hstack([np.ones((m, 1)), X])
    diff = np.dot(B , theta) - Y
    return (1/2*m) * np.dot(diff.transpose(), diff)


def gradient_function(theta, X, Y):
    B = np.hstack([np.ones((m, 1)), X])
    diff = np.dot(B, theta) - Y
    return (1/m) * np.dot(B.transpose(), diff)


def gradient_descent(X, Y, alpha, iter_num = 10000):
    theta = np.ones((n+1, 1)) # 初始
    J_ = gradient_function(theta, X, Y)
    num = 0
    while not all(abs(J_) <= 1e-5):     # 当梯度很小时，趋于平滑，最好是为0
        if num >= iter_num:
            break
        theta = theta - alpha * J_
        J_ = gradient_function(theta, X, Y)
        num += 1
    return theta


m = 100
n = 10
if __name__ == "__main__":
    np.random.seed(222)
    X = np.random.randn(m, n)
    Y = 10 * np.ones((m, 1))
    for i in range(n):
        Y += ((i+1) * X[:, i]).reshape(m, 1)

    alpha = 0.01    # 学习率
    theta = gradient_descent(X, Y, alpha)
    print('优化后的参数为：', theta)
    print("损失函数值为：", cost_function(theta, X, Y))