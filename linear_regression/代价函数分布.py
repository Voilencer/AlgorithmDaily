import numpy as np
import matplotlib.pyplot as plt
import random



def cost_function1(theta, X, Y):
    # y = k * x
    B = X
    diff = np.dot(B , theta) - Y
    return (1/2*m) * np.dot(diff.transpose(), diff)


def cost_function2(theta, X, Y):
    #  y = k * x + b
    B = np.hstack([np.ones((m, 1)), X])
    diff = np.dot(B , theta) - Y
    return (1/2*m) * np.dot(diff.transpose(), diff)


def xrange(start, end, step):
    assert step != 0
    val_max = max(start, end)
    val_min = min(start, end)
    res = []
    while val_min < val_max:
        res.append(start)
        start += step
        val_min += abs(step)
    return res


m = 100
if __name__ == "__main__":
    np.random.seed(222)

    ###################
    # X = 2 * np.random.randn(m, 1)
    # Y = 5 * X  #  +  np.random.randn(m, 1)
    #
    # l_theta, l_cost = [], []
    # for i in xrange(0, 10, 0.05):
    #     l_theta.append(i)
    #     l_cost.append(cost_function1(i, X, Y))
    # fig = plt.figure()
    # plt.scatter(l_theta, l_cost)
    # plt.title('y = kx + b')
    # plt.show()

    ##########################

    np.random.seed(222)
    X = 2 * np.random.randn(m, 1)
    Y = 5 * X + 5

    l_k, l_b, l_cost = [], [], []
    for i in xrange(0, 10, 0.1):
        for j in xrange(0, 10, 0.1):
            l_k.append(i)
            l_b.append(j)
            theta = np.array([i, j]).reshape((2,1))
            # print(cost_function2(theta, X, Y))
            l_cost.append(cost_function2(theta, X, Y)[0][0])

    fig = plt.figure()
    plt.figure()  # 设置画布大小
    ax = plt.axes(projection='3d')  # 设置三维轴
    ax.scatter3D(l_k, l_b, l_cost)  # 三个数组对应三个维度（三个数组中的数一一对应）
    plt.xlabel('X')
    plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
    plt.show()