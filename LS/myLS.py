import numpy as np
import random
import matplotlib.pyplot as plt



def cal_model(X, Y):
    # y = k*x + b  最小二乘法
    Bval, Lval = [], []
    for _x, _y in zip(X, Y):
        Bval.append([_x, 1])
        Lval.append([_y])
    Bmat = np.matrix(Bval)
    Lmat = np.matrix(Lval)
    gRes = (Bmat.T * Bmat).I * (Bmat.T * Lmat)
    Vmat = Bmat * gRes - Lmat
    sigma = np.sqrt(Vmat.T * Vmat / (len(X) - 2))
    return gRes.tolist(), sigma


def cal_model_error(para, x, y):
    y_gj = para[0][0] * x + para[1][0]
    return y_gj - y



if __name__ == "__main__":
    X = range(0, 100, 10)
    Y = [5 * _+  random.randint(0, 100) for _ in X]

    para, sigma =  cal_model(X, Y)
    print(para, sigma)

    x1 = 0
    y1 = para[0][0] * x1 + para[1][0]
    x2 = 90
    y2 = para[0][0] * x2 + para[1][0]

    fig = plt.figure()
    plt.scatter(X, Y)
    plt.plot([x1, x2], [y1, y2], color='r')
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title('LS直线拟合')
    plt.show()




