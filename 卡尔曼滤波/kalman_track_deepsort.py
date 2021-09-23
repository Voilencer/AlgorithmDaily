# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

"""
deepsort 跟踪 里面的卡尔曼滤波
物体匀速情况下，线性系统

Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""



class KalmanFilter(object):
    """
    (1) 8维状态量：x, y,  中心点位置     a,  宽高比     h,  高度  vx,     vy,     va,     vh
    (2) 物体匀速运动情况下
    (3) bbox是直接观测量
    """
    def __init__(self, ndim=4, dt=1):
        # 创建卡尔曼滤波模型矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt  # 状态预测更新矩阵
        self._update_mat = np.eye(ndim, 2 * ndim)  # 状态转移矩阵
        # 当前状态 运动和观测的不确定性
        self._std_weight_position = 1. / 20     # 位置
        self._std_weight_velocity = 1. / 160    # 速度

    def initiate(self, measurement):
        """ 从不关联的测量值创建跟踪
        观测量 :
            x, y, a, h
        返回值
            平均向量8维
            新跟踪的协方差矩阵（未观测到的速度初始化为0平均值）
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos) # 速度初始化为0
        mean = np.r_[mean_pos, mean_vel] # x_k = [p, v]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std)) # R 观测噪声协方差矩阵
        return mean, covariance

    def predict(self, mean, covariance):
        """预测
        参数：
            mean : ndarray
                物体状态的8维平均向量，先验估计
            covariance : ndarray
                物体状态的8*8协方差矩阵，先验估计
        返回
            预测向量
            预测协方差阵（未观测到的速度初始化为0均值）
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # ++
        # 过程噪声协方差矩阵
        mean = np.dot(self._motion_mat, mean) # x_k_minus = F * x_k_hat
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov # P_k = F * P_k-1 * F^T + Q
        return mean, covariance

    def project(self, mean, covariance):
        """状态分布投影到观测空间,状态转移，状态量->观测量
        参数：
            mean : 状态的8维均值向量
            covariance : 状态的8x8协方差矩阵
        返回：
            给定状态估计量投影后的均值和协方差矩阵
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """校正
        参数：
            mean : ndarray
                预测状态的平均向量(8维)
            covariance : ndarray
                预测状态的协方差矩阵(8x8)
            measurement : ndarray
                观测矩阵(x, y, a, h)
        返回：
            返回经过测量修正的状态分布
        """
        projected_mean, projected_cov = self.project(mean, covariance) # H * x_k_minus

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False) # Cholesky分解
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T # 计算卡尔曼增益
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T) # x_k_hat = x_k_minus + K(z - H * x_k_minus)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T)) # P_k
        return new_mean, new_covariance



if __name__ == "__main__":
    from utils import *
    my_kalman= KalmanFilter()

    path = r'./data/id_info/0.txt'
    info = read_path_id(path)
    info_ = []
    for ind, item in enumerate(info):
        print(ind)
        index = item['frame']
        l,t,r,b = item['bbox']
        x = (l+r)/2
        y = (t+b)/2
        a = (r-l) / (b - t)
        h = b - t

        measurement = np.array([x, y, a, h])
        if ind == 0:
            mean, covariance = my_kalman.initiate(measurement)
            x,y,a,h = measurement
        else:
            mean, covariance = my_kalman.predict(mean, covariance)
            new_mean, new_covariance = my_kalman.update(mean, covariance, measurement)
            mean = new_mean
            covariance = new_covariance
            x,y,a,h = [x for x in new_mean[:4]]
        l,t,r,b = int(x - a*h/2), int(y-h/2), int(x+a*h/2), int(y+h/2)
        info_.append({'frame':index, 'bbox':[l, t, r, b]})
        print('\t', info[ind]['bbox'])
        print('\t', info_[ind]['bbox'])

    # 预测
    # show_merge_info(info, info_)
    # mean, covariance = my_kalman.predict(mean, covariance)


    figure = plt.figure()
    for item1, item2 in zip(info, info_):
        type = 0
        plt.scatter([item1['frame']], [item1['bbox'][type]], color = 'r')
        plt.scatter([item1['frame']], [item1['bbox'][1]], color = 'g')
        plt.scatter([item1['frame']], [item1['bbox'][2]], color = 'b')
        plt.scatter([item1['frame']], [item1['bbox'][3]], color = 'y')
        # plt.scatter([item2['frame']], [item2['bbox'][type]], color = 'g')
    plt.show()

