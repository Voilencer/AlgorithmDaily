import numpy as np
import matplotlib.pyplot as plt


"""
吴恩达课堂作业，单变量线性回归
（1）梯度下降法
（2）正规方程方法（最小二乘）

"""

def show_data(data):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], marker='+', c='red')
    plt.show()


def show_res(theta, X, Y):
    x = np.linspace(min(X), max(X), 100)
    y = theta[0][0] + theta[1][0] * x
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, y, 'r', label='Prediction')
    ax.scatter(X, Y, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def show_cost(cost):
    fig = plt.figure()
    plt.plot(range(len(cost)), cost, c = 'r')
    plt.show()


def cost_function(theta, X, Y):
    # print(len(Y))
    return np.sum(np.power(np.dot(XX, theta) - Y, 2)) / (2 * len(Y))


def gradient_function(theta, X, Y):
    m = len(X)
    diff = np.dot(X, theta) - Y
    return (1/m) * np.dot(X.transpose(), diff)


def gradient_descent(X, Y, theta, alpha, iter_num = 1000):
    cost = [cost_function(theta, X, Y)]
    J_ = gradient_function(theta, X, Y)
    num = 0
    while not all(abs(J_) <= 1e-5):     # 当梯度很小时，趋于平滑，最好是为0
        if num >= iter_num:
            break
        theta = theta - alpha * J_
        cost.append(cost_function(theta, X, Y))
        J_ = gradient_function(theta, X, Y)
        num += 1
    return theta, cost


def normalize_feature(data):
    # 数据标准化，特征缩放
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)



if __name__ == "__main__":
    path = r'ex1data1.txt'
    data = np.loadtxt(path, delimiter=',')
    # print(data.shape)
    # show_data(data)

    data_ = normalize_feature(data)

    length  = len(data_[:,0])
    X = data_[:,0].reshape(length, 1)
    XX = np.hstack([np.ones((length, 1)), X])
    Y = data_[:,1].reshape(length, 1)
    alpha = 0.01
    iter = 1000
    theta = np.zeros((2, 1))

    theta, cost = gradient_descent(XX, Y, theta, alpha, iter)
    print(theta)

    show_res(theta, X, Y)
    show_cost(cost)


