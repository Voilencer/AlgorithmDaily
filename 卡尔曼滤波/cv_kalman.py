import cv2, re
import numpy as np
import matplotlib.pyplot as plt

"""
cv2.KalmanFilter(4, 2) 
    # 创建kalman滤波器 
    dynam_params：状态空间的维数； 4
    measure_param：测量值的维数；  2
    control_params：控制向量的维数，默认为0。由于这里该模型中并没有控制变量，因此也为0。
kalman.measurementMatrix 观测矩阵 H
kalman.transitionMatrix  状态转移矩阵 F
kalman.processNoiseCov 处理噪声协方差矩阵  Q
kalman.measurementNoiseCov 观测噪声协方差矩阵 R
kalman.controlMatrix 控制矩阵 B
kalman.statePost   校正状态 
kalman.statePre   预测状态
kalman.errorCovPost 后验方差协方差阵 P = (I-KH)P'(k)
kalman.errorCovPre 先验方差
kalman.gain 卡尔曼增益矩阵
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


path = r'./data/data2.txt'
info = read_data(path)

figure = plt.figure()
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.scatter(info['frame'], info['x'], color='r', label='x-观测值')
plt.scatter(info['frame'], info['y'], color='b', label='y-观测值')


# 状态量  x, y, Vx, Vy
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 设置测量矩阵 H
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 设置过程噪声协方差矩阵 Q
kalman.measurementNoiseCov = np.matrix([[1, 0], [0, 1]])    # 观测噪声方差

num = len(info['frame'])
for k in range(1, num):
    delt = (info['frame'][k] - info['frame'][k-1])
    kalman.transitionMatrix = np.array([[1, 0, delt, 0], [0, 1, 0, delt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 设置转移矩阵 F

    x, y = info['x'][k], info['y'][k]
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(current_measurement)  # 用当前测量来校正卡尔曼滤波器
    current_prediction = kalman.predict()
    plt.scatter(info['frame'][k], current_prediction[0][0], color='olive', label='x-预测值', marker='s')
    plt.scatter(info['frame'][k], current_prediction[1][0], color='pink', label='y-预测值', marker='*')
plt.legend(['x-观测值', 'y-观测值', 'x-预测值', 'y-预测值'])
plt.xlabel('frame')
plt.show()



if __name__ == "__main__":
    pass