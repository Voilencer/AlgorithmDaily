import numpy as np
import matplotlib.pyplot as plt
import re
import numpy as np

import cv2

"""
假设是匀速行驶

观测量:(x, y, v_x, v_y)


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

path = r'./data/data2.txt'
info = read_data(path)
num = len(info['frame'])

def get_kalman_filter():
    x_hat = np.matrix([[info['x'][0]], [info['y'][0]], [0], [0]])
    P = np.matrix(np.diag([1, 1, 1, 1]))
    Q = np.matrix(np.diag([1, 1, 1, 1]))
    # Q = np.matrix([[1/3, 0, 1/2, 0], [0, 1/3, 0, 1/2], [1/2, 0, 1, 0], [0, 1/2, 0, 1]], np.float32)
    H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = np.matrix([[1, 0], [0, 1]])
    figure = plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.scatter(info['frame'], info['x'], color='r', label='x-观测值')
    plt.scatter(info['frame'], info['y'], color='b', label='y-观测值')

    for k in range(1, num):
        delt_t = (info['frame'][k] - info['frame'][k - 1])
        F = np.matrix([[1, 0, delt_t, 0], [0, 1, 0, delt_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        # 预测
        x_hat_minus = F * x_hat
        P_minus = F * P * F.T + Q

        # 更新
        K = P_minus * H.T * np.linalg.inv(H * P_minus * H.T + R)
        x_hat = x_hat_minus + K * (np.matrix([[info['x'][k]], [info['y'][k]]]) - H * x_hat_minus)
        P = P_minus - K * H * P_minus

        plt.scatter(info['frame'][k], x_hat_minus.A[0][0], color='olive', label='x-预测值', marker='s')
        plt.scatter(info['frame'][k], x_hat_minus.A[1][0], color='pink', label='y-预测值', marker='*')

    # 往后预测10步
    for i in range(1, 11):
        # 多步预测
        delt_t = i
        F = np.matrix([[1, 0, delt_t, 0], [0, 1, 0, delt_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        x_hat_minus = F * x_hat
        plt.scatter(info['frame'][k] + i, x_hat_minus.A[0][0], color='olive', label='x-预测值')
        plt.scatter(info['frame'][k] + i, x_hat_minus.A[1][0], color='pink', label='y-预测值')
    plt.legend(['x-观测值', 'y-观测值', 'x-预测值', 'y-预测值'])
    plt.xlabel('frame')
    plt.show()



if __name__ == "__main__":
    get_kalman_filter()