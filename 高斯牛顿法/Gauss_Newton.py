import numpy as np
import matplotlib.pyplot as plt




def func(x, para):
    return para[0][0] * x**2 + para[1][0] * x

def jacobi(x, para):
    J = np.zeros((len(x), para.shape[0]))
    for i, item in enumerate(x):
        J[i, 0] = item ** 2
        J[i, 1] = item
    return np.matrix(J)



def fit_curve(x, y, iter_num = 10000, eps=1e-6):
    para = np.ones((2,1))  # (1)
    for i in range(iter_num):

        y_gj = func(x, para)
        error = y - y_gj # (2)
        J = jacobi(x, para)
        r = np.matrix(error)
        tmp = (J.T * J).I * J.T *  r # (3)
        para_ = para + tmp.A # (4)

        norm = np.sqrt(np.sum(tmp.A ** 2))

        if norm < eps:
            break
        para = para_
    return para


def show(x, y, para=None):
    figure = plt.figure()
    plt.scatter(x, y)

    if para is not None:
        x_ = np.arange(np.min(x), np.max(x), 0.05)
        y_ = para[0][0] * x_ **2 + para[1][0] * x_
        plt.plot(x_, y_, color='r')
    plt.show()



if __name__ == "__main__":
    x = np.random.randn(20, 1)
    mean = 0
    var = 0.5
    noise = np.random.normal(0, 1 ** 0.5, (20, 1))
    y = 3 * x**2 + 4 * x + noise

    para = fit_curve(x, y)
    show(x, y, para)







