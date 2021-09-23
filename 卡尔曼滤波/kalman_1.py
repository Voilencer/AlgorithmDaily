import numpy as np
import matplotlib.pyplot as plt



class kalman:
    def __init__(self):
        # x, y
        self.F = np.matrix([[1, 1], [0, 1]])  # 状态转移矩阵
        self.P = np.matrix([[1, 0], [0, 1]])  # 状态协方差矩阵
        self.Q = np.matrix([[0.00001, 0], [0, 0.00001]])  # 状态转移协方差
        self.H = np.matrix([1, 0])  # 观测矩阵 Z_t = Hx_t + v
        self.R = 1  # 观测噪声方差

    def init(self, val):
        self.x_hat = np.matrix([[val], [0]])  # 初始状态

    def predict(self):
        # 预测
        self.x_hat_minus = self.F * self.x_hat
        self.P_minus = self.F * self.P * self.F.T + self.Q

    def update(self, measure):
        # 更新
        self.K = self.P_minus * self.H.T * np.linalg.inv(self.H * self.P_minus * self.H.T + self.R)
        self.x_hat = self.x_hat_minus + self.K * (measure - self.H * self.x_hat_minus)
        self.P = self.P_minus - self.K * self.H * self.P_minus



if __name__ == "__main__":
    num = 100
    Z = np.arange(1, num + 1, 1)
    noise = 10 * np.random.randn(num)  # 均值为0，方差为2的观测噪声,标准正态分布
    Z = Z + noise  # 观测值+噪声

    my_kalman = kalman()
    my_kalman.init(Z[0])

    figure = plt.figure()
    plt.scatter(range(num), Z, color='b')
    for k in range(1, num):
        my_kalman.predict()
        my_kalman.update(Z[k])
        plt.scatter(k, my_kalman.x_hat.A[0][0], color='r')
    plt.show()