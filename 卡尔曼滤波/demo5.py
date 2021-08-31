import numpy as np
import matplotlib.pyplot as plt



"""匀速运动



"""


num = 100
Z = np.arange(1, num+1, 1)
noise = 10 * np.random.randn(num)  # 均值为0，方差为2的观测噪声,标准正态分布
Z = Z + noise  # 观测值+噪声

figure = plt.figure()
plt.scatter(range(num), Z, color='b')

x = np.matrix([[0], [0]]) # 初始状态
F = np.matrix([[1, 1], [0, 1]]) # 状态转移矩阵
P = np.matrix([[1, 0], [0, 1]]) # 状态协方差矩阵
Q = np.matrix([[0.00001, 0], [0, 0.00001]]) # 状态转移协方差
H = np.matrix([1, 0])  # 观测矩阵 Z_t = Hx_t + v
R = 1   # 观测噪声方差

for k in range(1, num):
    # 预测
    x_hat_minus = F * x
    P_minus = F * P * F.T + Q

    # 更新
    K = P_minus * H.T * np.linalg.inv(H * P_minus * H.T + R)
    x = x_hat_minus + K * (Z[k] - H * x_hat_minus)
    P = P_minus - K * H * P_minus

    plt.scatter(k, x.A[0][0], color='r')
plt.show()



if __name__ == "__main__":
    pass