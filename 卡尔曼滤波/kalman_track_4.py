import numpy as np
import matplotlib.pyplot as plt
import re, cv2
from utils import *



class KalmanFilter:
    """
    目标在图像上匀速运动
    状态量： l,t,r,b, vl,vt,vr,vb
    观测量： l,t,r,b
    观测噪声方差和过程噪声误差和bbox高度有关,h越大，误差越大，实时更新
    """
    def __init__(self):
        self._std_weight_pos = 1. / 10     # 位置
        self._std_weight_vel = 1. / 100    # 速度
        self.H = np.matrix(np.zeros((4, 8)))  # 观测矩阵 Z_t = Hx_t + v
        for i in range(4):
            self.H[i, i] = 1
        self.F = np.matrix(np.eye(8)) # 状态转移矩阵

    def init(self, val):
        l,t,r,b = val
        mean_pos = np.matrix([[l], [t], [r], [b]])
        mean_vel = np.zeros_like(mean_pos) # 速度初始化为0
        self.x_hat = np.r_[mean_pos, mean_vel] # x_k = [p, v]
        h = b - t
        std_pos = [
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            self._std_weight_pos * h]
        std_vel = [
            self._std_weight_vel * h,
            self._std_weight_vel * h,
            self._std_weight_vel * h,
            self._std_weight_vel * h]
        self.P = np.diag(np.square(np.r_[std_pos, std_vel])) # 状态协方差矩阵

    def predict(self, delt_t, val=None):
        if val is not None:
            h = val[3] - val[1]
            std_pos = [
                self._std_weight_pos * h,
                self._std_weight_pos * h,
                self._std_weight_pos * h,
                self._std_weight_pos * h]
            std_vel = [
                self._std_weight_vel * h,
                self._std_weight_vel * h,
                self._std_weight_vel * h,
                self._std_weight_vel * h]
            self.Q = np.diag(np.square(np.r_[std_pos, std_vel])) # 过程噪声实时变化
        for i in range(4):
            self.F[i, i+4] = delt_t
        self.x_hat_minus = self.F * self.x_hat
        self.P_minus = self.F * self.P * self.F.T + self.Q

    def update(self, val):
        l,t,r,b = val
        h = b - t
        std_pos = [
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            self._std_weight_pos * h]
        self.R = np.diag(np.square(std_pos)) # 观测噪声方差
        measure = np.matrix([[l], [t], [r], [b]])
        self.K = self.P_minus * self.H.T * np.linalg.inv(self.H * self.P_minus * self.H.T + self.R)
        self.x_hat = self.x_hat_minus + self.K * (measure - self.H * self.x_hat_minus)
        self.P = self.P_minus - self.K * self.H * self.P_minus



if __name__ == "__main__":
    path_id = r'./data/id_info/1.txt'
    info = read_path_id(path_id)
    num = len(info)
    # show_bbox_by_info(info)

    fig = plt.figure()

    kalman = KalmanFilter()
    kalman.init(info[0]['bbox'])
    for i in range(1, num-10):
        # predict
        delt_t = info[i]['frame'] - info[i-1]['frame']
        kalman.predict(delt_t, info[i]['bbox'])
        # update
        kalman.update(info[i]['bbox'])
        for m in range(4):
            plt.subplot(2, 2, m+1);   plt.scatter(info[i]['frame'], info[i]['bbox'][m],  color='r', marker='s')
            plt.subplot(2, 2, m+1);   plt.scatter(info[i]['frame'], kalman.x_hat.A[m][0], color='g', marker='*')
    # 预测后10帧
    for j in range(1, 11):
        kalman.predict(j)
        for m in range(4):
            plt.subplot(2, 2, m+1);   plt.scatter(info[i+j]['frame'], info[i+j]['bbox'][m],  color='r', marker='s')
            plt.subplot(2, 2, m+1);   plt.scatter(info[i+j]['frame'], kalman.x_hat_minus.A[m][0], color='g', marker='*')
    plt.subplot(2,2,1);plt.legend(['l', 'l-KF'])
    plt.subplot(2,2,2);plt.legend(['t', 't-KF'])
    plt.subplot(2,2,3);plt.legend(['r', 'r-KF'])
    plt.subplot(2,2,4);plt.legend(['b', 'b-KF'])
    plt.show()



