import numpy as np
import matplotlib.pyplot as plt
import re, cv2
import numpy as np



"""
状态量：(x, y, v_x, v_y)
观测量：(x, y, v_x, v_y)
匀速运动
"""



def read_data(path):
    info = {'frame':[], 'x':[], 'y':[]}
    with open(path, 'r') as fp:
        for item in fp.readlines():
            tmp = re.split(r'\[|\]|,', item.split('\n')[0])
            info['frame'].append(int(tmp[1]))
            info['x'].append(int(tmp[3]))
            info['y'].append(int(tmp[4]))
    return info


def show(info):
    figure = plt.figure()
    plt.scatter(info['frame'], info['x'], color='r')
    plt.scatter(info['frame'], info['y'], color='b')
    plt.show()



class kalman_1:
    def __init__(self):
        # x, y, Vx, Vy
        self.std_pos = 1 / 20
        self.std_vel = 1 / 160
        self.F = np.matrix([[1, 1], [0, 1]])  # 状态转移矩阵
        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

    def init(self, x, y):
        self.x_hat = np.matrix([[x], [y], [0], [0]])
        std = np.square([self.std_pos * 300,
                         self.std_pos * 300,
                         self.std_vel * 300,
                         self.std_vel * 300])
        self.Q = np.matrix(np.diag(std))  # 状态转移协方差
        self.P = np.matrix(np.diag(std)) # 状态协方差矩阵
        self.R = np.matrix(np.diag(np.square([self.std_pos * 300,
                                              self.std_pos * 300]))) # 观测噪声方差

    def predict(self, delt_t):
        # 预测
        self.F = np.matrix([[1, 0, delt_t, 0], [0, 1, 0, delt_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.x_hat_minus = self.F * self.x_hat
        self.P_minus = self.F * self.P * self.F.T + self.Q

    def update(self, x, y):
        # 更新
        measure = np.matrix([[x], [y]])
        self.K = self.P_minus * self.H.T * np.linalg.inv(self.H * self.P_minus * self.H.T + self.R)
        self.x_hat = self.x_hat_minus + self.K * (measure - self.H * self.x_hat_minus)
        self.P = self.P_minus - self.K * self.H * self.P_minus



class kalman_2:
    def __init__(self):
        # x, y, Vx, Vy, Ax, Ay
        self.std_pos = 1 / 20
        self.std_vel = 1 / 160
        self.F = np.matrix([[1, 1], [0, 1]])  # 状态转移矩阵
        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

    def init(self, x, y):
        self.x_hat = np.matrix([[x], [y], [0], [0]])
        std = np.square([self.std_pos * 300,
                         self.std_pos * 300,
                         self.std_vel * 300,
                         self.std_vel * 300])
        self.Q = np.matrix(np.diag(std))  # 状态转移协方差
        self.P = np.matrix(np.diag(std)) # 状态协方差矩阵
        self.R = np.matrix(np.diag(np.square([self.std_pos * 300,
                                              self.std_pos * 300]))) # 观测噪声方差

    def predict(self, delt_t):
        # 预测
        self.F = np.matrix([[1, 0, delt_t, 0], [0, 1, 0, delt_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.x_hat_minus = self.F * self.x_hat
        self.P_minus = self.F * self.P * self.F.T + self.Q

    def update(self, x, y):
        # 更新
        measure = np.matrix([[x], [y]])
        self.K = self.P_minus * self.H.T * np.linalg.inv(self.H * self.P_minus * self.H.T + self.R)
        self.x_hat = self.x_hat_minus + self.K * (measure - self.H * self.x_hat_minus)
        self.P = self.P_minus - self.K * self.H * self.P_minus




if __name__ == "__main__":
    path = r'./data/frame_x_y_2.txt'
    info = read_data(path)
    num = len(info['frame'])

    figure = plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(info['frame'], info['x'], color='r', label='x-观测值')
    plt.scatter(info['frame'], info['y'], color='b', label='y-观测值')

    my_kalman_1 = kalman_1()
    my_kalman_1.init(info['x'][0], info['y'][0])
    for i in range(1, num):
        delt_t = info['frame'][i] - info['frame'][i-1]
        my_kalman_1.predict(delt_t)
        my_kalman_1.update(info['x'][i], info['y'][i])

        plt.scatter(info['frame'][i], my_kalman_1.x_hat_minus.A[0][0], color='olive', label='x-预测值', marker='s')
        plt.scatter(info['frame'][i], my_kalman_1.x_hat_minus.A[1][0], color='pink', label='y-预测值', marker='*')
    plt.legend(['x-观测值', 'y-观测值', 'x-预测值', 'y-预测值'])
    plt.xlabel('frame')
    plt.title('kalman_filter')
    plt.show()