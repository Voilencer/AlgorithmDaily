import numpy as np
import matplotlib.pyplot as plt
import re, cv2
from utils import *



class KalmanFilter:
    """
    状态量： l,t,r,b, vl,vt,vr,vb
    观测量： l,t,r,b
    观测噪声方差和过程噪声误差和bbox高度有关,h越大，误差越大，实时更新
    标定图像和世界坐标系之间的单应变换矩阵：
        近似标定，选取几组对应点，存在误差
        [70.94, -29.7554, 1117.79, -2.00005, 2.58351, 381.313, -0.000604969, -0.027283, 1]
    目标在世界坐标系内匀速运动
    """
    def __init__(self, homography):
        self._std_weight_pos = 1. / 10     # 位置
        self._std_weight_vel = 1. / 100    # 速度
        self.H = np.matrix(np.zeros((4, 8)))  # 观测矩阵 Z_t = Hx_t + v
        for i in range(4):
            self.H[i, i] = 1
        self.F = np.matrix(np.eye(8)) # 状态转移矩阵
        self.homography = np.reshape(homography, (3,3))

    def _img2world(self, val):
        l,t,r,b = val
        tmp = np.dot(np.linalg.inv(self.homography), np.array([[l], [t], [1]]))
        l_ = tmp[0] / tmp[2]
        t_ = tmp[1] / tmp[2]
        tmp = np.dot(np.linalg.inv(self.homography), np.array([[r], [b], [1]]))
        r_ = tmp[0] / tmp[2]
        b_ = tmp[1] / tmp[2]
        return [l_, t_, r_, b_]


    def _world2img(self, val):
        l,t,r,b = val
        tmp = np.dot(self.homography, np.array([[l], [t], [1]]))
        l_ = tmp[0] / tmp[2]
        t_ = tmp[1] / tmp[2]
        tmp = np.dot(self.homography, np.array([[r], [b], [1]]))
        r_ = tmp[0] / tmp[2]
        b_ = tmp[1] / tmp[2]
        return [l_, t_, r_, b_]

    def init(self, val):
        ltrb = self._img2world(val)
        mean_pos = np.matrix(ltrb)
        mean_vel = np.zeros_like(mean_pos) # 速度初始化为0
        self.x_hat = np.r_[mean_pos, mean_vel] # x_k = [p, v]
        h = val[3]-val[1]
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
        ltrb = self._img2world(val)
        self.R = np.diag(np.square(std_pos)) # 观测噪声方差
        measure = np.matrix(ltrb)
        self.K = self.P_minus * self.H.T * np.linalg.inv(self.H * self.P_minus * self.H.T + self.R)
        self.x_hat = self.x_hat_minus + self.K * (measure - self.H * self.x_hat_minus)
        self.P = self.P_minus - self.K * self.H * self.P_minus



if __name__ == "__main__":
    path_id = r'./data/id_info/1.txt'
    info = read_path_id(path_id)
    num = len(info)
    # show_bbox_by_info(info)

    fig = plt.figure()
    homography = [70.94, -29.7554, 1117.79, -2.00005, 2.58351, 381.313, -0.000604969, -0.027283, 1]
    kalman = KalmanFilter(homography)
    kalman.init(info[0]['bbox'])
    for i in range(1, num-10):
        # predict
        delt_t = info[i]['frame'] - info[i-1]['frame']
        kalman.predict(delt_t, info[i]['bbox'])
        # update
        kalman.update(info[i]['bbox'])
        val = kalman._world2img(kalman.x_hat.A[:4, 0])
        for m in range(4):
            plt.subplot(2, 2, m+1);   plt.scatter(info[i]['frame'], info[i]['bbox'][m],  color='r', marker='s')
            plt.subplot(2, 2, m+1);   plt.scatter(info[i]['frame'], val[m], color='g', marker='*')
    # 预测后10帧
    for j in range(1, 11):
        kalman.predict(j)
        val = kalman._world2img(kalman.x_hat_minus.A[:4, 0])
        for m in range(4):
            plt.subplot(2, 2, m+1);   plt.scatter(info[i+j]['frame'], info[i+j]['bbox'][m],  color='r', marker='s')
            plt.subplot(2, 2, m+1);   plt.scatter(info[i+j]['frame'], val[m], color='g', marker='*')
    plt.subplot(2,2,1);plt.legend(['l', 'l-KF'])
    plt.subplot(2,2,2);plt.legend(['t', 't-KF'])
    plt.subplot(2,2,3);plt.legend(['r', 'r-KF'])
    plt.subplot(2,2,4);plt.legend(['b', 'b-KF'])
    plt.show()



