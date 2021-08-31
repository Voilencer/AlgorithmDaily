import numpy as np
import matplotlib.pyplot as plt



""" 二元非线性回归，拟合二次曲线
模型函数：y = ax^2 + bx
代价函数：


"""


def cost_function(para, X, Y):
    return np.sum(para[0] * X**2 + para[1] * X - Y ) / (2 * m)


def  gradient_function(para, X, Y):
    B = np.hstack([X**2, X])
    diff = np.dot(B, para) - Y
    return (1/m) * np.dot(B.transpose(), diff)


def gradient_descent(X, Y, alpha, iter_num=10000):
    para = np.ones((2,1))
    J_ = gradient_function(para, X, Y) # 求梯度
    num = 0
    while not all(abs(J_) <= 1e-6):
        print(num, iter_num)
        if num >= iter_num:  # 限制迭代次数，防止一致不收敛
            break
        para = para - alpha * J_  # 梯度下降
        J_ = gradient_function(para, X, Y)
        num += 1
    return para



m = 100
if __name__ == "__main__":
    ##### 二元线性回归
    np.random.seed(222)

    X = 2 * np.random.randn(m, 1)
    Y = 5 * X **2  +  4 * X + 3 * np.random.randn(m, 1)

    alpha = 0.001    #学习率
    para = gradient_descent(X, Y, alpha)
    print('优化后的参数： a=%.3f b = %.3f' % (para[0], para[1]))
    print("损失函数值为：", cost_function(para, X, Y))

    fig = plt.figure()
    plt.scatter(X, Y)
    X_ = np.arange(np.min(X), np.max(X), 0.1)
    Y_ = para[0] * X_**2 + para[1] * X_
    plt.plot(X_, Y_, color = 'red')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title('梯度下降法')
    plt.show()

