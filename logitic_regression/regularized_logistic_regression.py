import numpy as np
import matplotlib.pyplot as plt



"""逻辑回归

假设方程：h_θ(x) = g(θt * X)
          g(z) = 1 / (1 + e^(-z))
代价函数：

优化方法：梯度下降法

"""


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def cost_function(theta, X, Y, ambda=1):
    X = np.matrix(X)
    Y = np.matrix(Y)
    Theta = np.matrix(theta)
    first = np.multiply(-Y, np.log(sigmoid(X * Theta)))
    second = np.multiply(1-Y, np.log(1 - sigmoid(X * Theta)))
    return np.sum(first - second) / m


def regularized_cost_function(theta, X, Y, ambda=1):
    m = X.shape[0]
    regularized_term = (ambda/2*m) * np.sum(np.power(theta, 2))
    return regularized_term + cost_function(theta, X, Y, ambda)


def gradient_function(theta, X, Y):
    diff = sigmoid(np.dot(X, theta)) - Y
    return (np.dot(X.T, diff)) / m


def regularized_gradient_function(theta, X, Y, ambda=1):
    m = X.shape[0]
    regularized_item = np.vstack([[0], theta[1:] * ambda / m])
    return gradient_function(theta, X, Y) + regularized_item



def gradient_descent(X, Y, theta, alpha, iter_num = 10000, ambda=1):
    gradient = regularized_gradient_function(theta, X, Y, ambda)
    num = 0
    while not all(abs(gradient) < 1e-6):
        if num > iter_num:
            break
        theta = theta - alpha * gradient
        gradient = regularized_gradient_function(theta, X, Y, ambda)
        num += 1
    return theta


def show_data(data):
    ind_one = np.where(data[:, 2] == 1)
    ind_zero = np.where(data[:, 2] == 0)
    fig = plt.figure()
    plt.scatter(data[ind_one,0], data[ind_one,1], marker='+', c='red', label='1')
    plt.scatter(data[ind_zero, 0], data[ind_zero, 1], marker='o', c='blue', label='0')
    plt.legend()
    plt.show()


def show_res():
    pass


def predict(x, theta):
    B = np.hstack([1, x])
    z = np.dot(theta.T, B)
    return sigmoid(z)




if __name__ == "__main__":
    path = 'ex2data1.txt'
    data = np.loadtxt(path, delimiter=',')
    show_data(data)

    m = data.shape[0]
    X = np.hstack([np.ones((m, 1)), data[:, :2]])
    Y = np.mat(data[:, 2]).reshape(m, 1)
    theta = np.ones((3, 1))
    alpha = 0.001
    iter = 1000

    res = gradient_descent(X, Y, theta, alpha, iter)
    print('optim', res)
    print('cost function', cost_function(res, X, Y))

    # predcit
    X0 = X[-1, 1:3]
    predict(X0, res)