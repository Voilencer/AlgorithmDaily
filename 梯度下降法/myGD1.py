import numpy as np
import random
import matplotlib.pyplot as plt



""" 一元线性回归，拟合直线
模型函数： h(x) = θ0 + θ1 * x

代价函数 J(θ) = 1/2m sum(（h(i)-y(i)）^2)
    J'(θ) = [J'(θ0) J'(θ1)]

梯度下降法：   θ = θ0 - learning_rate * J'(θ)
    可以用梯度值大小作为迭代终止条件

"""


def cost_function(theta, X, Y):
    return np.sum(theta[0] + theta[1] * X - Y) / (2*m)


def gradient_function(theta, X, Y):
    B = np.hstack([np.ones((m, 1)), X])
    diff = np.dot(B, theta) - Y
    return (1/m) * np.dot(B.transpose(), diff)


def gradient_descent(X, Y, alpha, iter_num = 1000):
    theta = np.ones((2, 1)) # 初始值
    J_ = gradient_function(theta, X, Y) # 求梯度
    num = 0
    while not all(abs(J_) <= 1e-5):     # 当梯度很小时，趋于平滑，最好是为0
        if num >= iter_num: # 限制迭代次数，防止一致不收敛
            break
        theta = theta - alpha * J_   # 梯度下降
        J_ = gradient_function(theta, X, Y)
        num += 1
    return theta



m = 100
if __name__ == "__main__":
    ##### 一元线性回归
    np.random.seed(222)
    X = 2 * np.random.randn(m, 1)
    Y = 5 * X +  np.random.randn(m, 1)

    alpha = 0.01    #学习率
    theta = gradient_descent(X, Y, alpha)
    print('优化后的参数为：', theta)
    print("损失函数值为：", cost_function(theta, X, Y))

    fig = plt.figure()
    plt.scatter(X, Y)

    X_ = np.arange(np.min(X), np.max(X), 0.1)
    Y_ = theta[0] + theta[1] * X_
    plt.plot(X_, Y_, color = 'red')

    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.title('梯度下降法')
    plt.show()