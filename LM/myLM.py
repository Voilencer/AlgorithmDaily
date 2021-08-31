import numpy as np
import matplotlib.pyplot as plt





def show(x, y, para=None):
    figure = plt.figure()
    plt.scatter(x, y)

    if para is not None:
        x_ = np.arange(np.min(x), np.max(x), 0.05)
        y_ = para[0][0] * np.exp(para[1][0] / (x_ + para[2][0]))
        plt.plot(x_, y_, color='r')
    plt.show()


def func(x, para):
    # y = k1 * exp(k2 / (x + k3))
    return para[0][0] * np.exp(para[1][0] / (x + para[2][0]))


def jacobi(x, para):
    J = np.zeros((len(x), para.shape[0]))
    for i, item in enumerate(x):
        J[i, 0] = np.exp(para[1][0] / (item + para[2][0]))
        J[i, 1] = para[1][0] * np.exp(para[1][0] / (item + para[2][0])) / (item + para[2][0])
        J[i, 2] = para[1][0] * np.exp(para[1][0] / (item + para[2][0])) * (-para[1][0] / (item + para[2][0])**2)
    return np.matrix(J)


def fit_curve(x, y, iter_num = 10000, eps=1e-8):
    num_paras = 3
    para_past = np.ones((num_paras,1))  # 参数初始化
    y_gj = func(x, para_past)
    J = jacobi(x, para_past)  # jacobi
    r_past = np.matrix(y - y_gj).T  # 残差矩阵
    print(J.shape, r_past.shape)
    g = J.T * r_past

    tao = 1e-3 # (1e-8, 1)
    u = tao * np.max(J.T * J) # 阻尼因子初始化值
    v = 2

    norm_inf = np.linalg.norm(J.T * r_past, ord = np.inf)
    stop = norm_inf < eps

    num = 0
    while (not stop) and num <iter_num:
        num += 1
        while True:
            H_lm = J.T * J + u * np.eye(num_paras)
            delt = H_lm.I * g
            norm_2 = np.linalg.norm(delt)
            if norm_2 < eps:
                stop = True
            else:
                para_cur = para_past + delt.A   # 更新参数
                y_gj_cur = func(x, para_cur)
                J_cur = jacobi(x, para_cur)
                r_cur = np.matrix(y - y_gj_cur).T
                rou = ((np.linalg.norm(r_past) ** 2 - np.linalg.norm(r_cur) ** 2) / (delt.T.dot(u * delt + g))).A[0][0]
                if rou > 0:
                    para_past = para_cur
                    r_past = r_cur
                    J = jacobi(x, para_past)
                    g = J.T * r_past
                    stop  = (np.linalg.norm(g, ord=np.inf) <= eps) or (np.linalg.norm(r_past)**2 <= eps)
                    u = u * max(1 / 3, 1 - (2 * rou - 1) ** 3)
                    v = 2
                else:
                    u *= v
                    v *= 2
            if rou > 0 or stop:
                break
    return para_past




if __name__ == "__main__":
    # y = k1 * exp(k2 / (x + k3))
    x = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.15]
    y = [2.98, 3.06, 3.17, 3.39, 3.71, 4.17, 4.98, 6.41, 9.09, 15.73, 57.38, 49.78, 42.42, 35.34, 29.87,
                      24.94, 18.71, 11.49]

    paras = fit_curve(x, y)
    print(paras)

    show(x, y, paras)
