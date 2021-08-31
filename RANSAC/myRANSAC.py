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
    return gRes, sigma


def estimate_error(para, x, y):
    # kx+b-y
    y_gj = para[0][0] * x + para[1][0]
    return y_gj - y


def myRANSAC(X, Y, inner_thr, iter_num, model_num):
    npts = len(X)
    l_inner_num = []
    l_para, l_sigma = [], []
    for i in range(iter_num):
        inner_ind = []
        rand_ind = random.sample(range(npts), model_num)
        inner_ind.extend(rand_ind)
        not_rand_ind = [x for x in range(npts) if x not in rand_ind]
        ptsP_, ptsR_ = [], []
        for j in rand_ind:
            ptsP_.append(X[j])
            ptsR_.append(Y[j])
        para, sigma = cal_model(ptsP_, ptsR_)
        inner_num = 0
        for k in not_rand_ind:
            sigma = estimate_error(para, X[k], Y[k])
            if sigma[0][0] < inner_thr:
                inner_num += 1
                inner_ind.append(k)
        l_inner_num.append(inner_num)
        l_para.append(para)
        l_sigma.append(sigma)
    max_ind = np.argmax(l_inner_num)
    return l_para[max_ind].tolist(), l_sigma[max_ind]




if __name__ == "__main__":
    X = range(0, 100, 10)
    Y = [5 * _+  random.randint(0, 100) for _ in X]

    inner_thr = 10   # 阈值
    iter_num = 100   # 迭代次数
    model_num = 4   # 模型最小点数

    para, sigma =  myRANSAC(X, Y, inner_thr, iter_num, model_num)
    print(para, sigma)

    x1 = 0
    y1 = para[0][0] * x1 + para[1][0]
    x2 = 90
    y2 = para[0][0] * x2 + para[1][0]

    fig = plt.figure()
    plt.scatter(X, Y)
    plt.plot([x1, x2], [y1, y2], color='r')
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title('RANSAC直线拟合')
    plt.show()




