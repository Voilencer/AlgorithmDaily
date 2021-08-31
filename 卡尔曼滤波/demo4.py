import numpy as np
import matplotlib.pyplot as plt



# 模拟数据
t = np.linspace(1, 100, 100)
a = 0.5
position = (a * t ** 2) / 2

position_noise = position + np.random.normal(0, 120, size=(t.shape[0]))

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.plot(t, position, label='理想值')
plt.plot(t, position_noise, label='观测值')

# 初始的估计导弹的位置就直接用GPS测量的位置
predicts = [position_noise[0]]
position_predict = predicts[0]

predict_var = 0
odo_var = 120 ** 2  # 自己设定的位置测量仪器的方差，越大则测量值占比越低
v_std = 50  # 速度测量仪器的方差
for i in range(1, t.shape[0]):
    # 预测
    dv = (position[i] - position[i - 1]) + np.random.normal(0, 50)  # 模拟从IMU读取出的速度
    position_predict = position_predict + dv  # 利用上个时刻的位置和速度预测当前位置
    predict_var += v_std ** 2  # 更新预测数据的方差

    # 更新：Kalman滤波
    position_predict = position_predict * odo_var / (predict_var + odo_var) + position_noise[i] * predict_var / (
                predict_var + odo_var)
    predict_var = (predict_var * odo_var) / (predict_var + odo_var) ** 2
    predicts.append(position_predict)

plt.plot(t, predicts, label='kalman滤波值')
plt.legend()
plt.show()
