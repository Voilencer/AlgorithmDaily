import numpy as np
import matplotlib.pyplot as plt


"""
吴恩达课堂作业，多变量线性回归
（1）梯度下降法
（2）正规方程方法（最小二乘）


"""



def show_data(data):
    # data[:,0] 大小     data[:,1] 数量     data[:,2] 价格
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.subplot(1,2,1); plt.scatter(data[:,0], data[:,2], marker='+', c='red');plt.title("大小—单价");
    plt.subplot(1,2,2); plt.scatter(data[:,1], data[:,2], marker='o', c='green');plt.title("卧室数量—单价");
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
    path = r'ex1data2.txt'
    data = np.loadtxt(path, delimiter=',')
    data_ = normalize_feature(data)

    length = len(data_[:,0])
    X = data_[:,0:2].reshape(length, 2)
    XX = np.hstack([np.ones((length, 1)), X])
    Y = data_[:,1].reshape(length, 1)
    alpha = 0.01
    iter = 500
    theta = np.zeros((3, 1))

    theta, cost = gradient_descent(XX, Y, theta, alpha, iter)
    print(theta)

    show_cost(cost)