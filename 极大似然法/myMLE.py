import numpy as np
import matplotlib.pyplot as plt


def MLE_Gaussian():
    # 正态分布 data~N(5, 4)
    N = 10000000
    np.random.seed(222)
    data = 2 * np.random.randn(N, 1) + 5

    plt.figure()
    plt.hist(data, 100)
    plt.show()

    miu_ = np.mean(data)
    sigma2_ = np.mean((data-miu_)**2)
    print("miu = %f, sigma = %f" % (miu_, sigma2_))



if __name__ == "__main__":

    MLE_Gaussian()








