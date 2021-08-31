[TOC]

***

# 概述

* 寻找参数向量，使得函数值最小的非线性优化算法
* `Levenberg-Marquardt`， LM算法，列文伯格-马夸尔特法
* `LM`算法成为**`Gauss-Newton`**算法与**`最速下降法(梯度下降法,GD)`**的结合,`μ`很小时，类似于高斯牛顿法；`μ`很大时，类似于LM法
* 阻尼因子`μ`
* 对于过参数化问题不敏感，能有效处理冗余参数问题，使代价函数陷入局部极小值的机会大大减小

# 对比

* 几种优化方法对比(转载)

| 方法         | 介绍                                                         | 迭代公式                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| 最速下降法   | 负梯度方向，收敛速度慢                                       | $x_{k+1}=x_k-\alpha g_k$                         |
| Newton 法    | 保留泰勒级数一阶和二阶项，二次收敛速度，但每步都计算Hessian矩阵，复杂 | $x_{k+1}=x_k-H^{-1}_{k}g_k$                      |
| 高斯牛顿法法 | 目标函数的Jacobian 矩阵近似H矩阵，提高算法效率，但H矩阵不满秩则无法迭代 | $x_{k+1}=x_k-(J^TJ)^{-1}J^Tr=x_k-(J^TJ)^{-1}g_k$ |
| LM法         | 信赖域算法，解决H矩阵不满秩或非正定                          | $x_{k+1}=x_k-(J^TJ+μI)^{-1}g_k$                  |

# 推导

* 目标函数优化，转化为最小化误差`f(x)`平方和

$$
minF(x_0+\triangle x) = ||f(x_0+\triangle x||^2
$$

* 一阶泰勒展开

$$
minF(x_0+\triangle x) = ||f(x_0)+J^T\triangle x||^2
$$

* 雅可比矩阵`J`

$$
J=\left[\begin{matrix}
\frac{∂y_1}{x_1} ... \frac{∂y_1}{x_n} \\
\cdots \\
\frac{∂y_m}{x_1} ... \frac{∂y_m}{x_n} \\
\end{matrix} \right]
$$

* 求极值，函数求导，导数为0

$$
\frac{∂||f(x_0)+J^T\triangle x||^2}{∂\triangle x} = 0
$$



* 求解得到

$$
JJ^T\triangle x = -Jf(x_0)
$$

* 迭代公式为

$$
\triangle x = -(J^TJ)^{-1}J^Tf(x_0)
$$

* `J^TJ`不一定可逆，需要引入单位阵

$$
H≈J^TJ+μI
$$

* 更新迭代公式为

$$
x_{k+1}=x_k-(J^TJ+μI)^{-1}g_k
$$



# 应用

* 实现可以参照论文里面的伪代码（完全一致）

<img src="C:\Users\Voilencer\AppData\Roaming\Typora\typora-user-images\image-20210819202640769.png" alt="image-20210819202640769" style="zoom:67%;" />

## 曲线拟合

* 一组数据

```
# y = k1 * exp(k2 / (x + k3))
x = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.15]
y = [2.98, 3.06, 3.17, 3.39, 3.71, 4.17, 4.98, 6.41, 9.09, 15.73, 57.38, 49.78, 42.42, 35.34, 29.87, 24.94, 18.71, 11.49]
```

* 曲线拟合

```
def func(x, para):
    # y = k1 * exp(k2 / (x + k3))
    return para[0][0] * np.exp(para[1][0] / (x + para[2][0]))

def jacobi(x, para):
    J = np.zeros((len(x), para.shape[0]))
    for i, item in enumerate(x):
        J[i, 0] = np.exp(para[1][0] / (item + para[2][0]))
        J[i, 1] = para[1][0] * np.exp(para[1][0] / (item + para[2][0])) / (item + para[2][0])
        J[i, 2] = para[1][0] * np.exp(para[1][0] / (item + para[2][0])) * (-para[1][0] / (item + para[2][0])**2)
    return np.matrix(J)
    
def fit_curve(x, y, iter_num = 10000, eps=1e-8):
    num_paras = 3
    para_past = np.ones((num_paras,1))  # 参数初始化
    y_gj = func(x, para_past)
    J = jacobi(x, para_past)  # jacobi
    r_past = np.matrix(y - y_gj).T  # 残差矩阵
    print(J.shape, r_past.shape)
    g = J.T * r_past

    tao = 1e-3 # (1e-8, 1)
    u = tao * np.max(J.T * J) # 阻尼因子初始化值
    v = 2

    norm_inf = np.linalg.norm(J.T * r_past, ord = np.inf)
    stop = norm_inf < eps

    num = 0
    while (not stop) and num <iter_num:
        num += 1
        while True:
            H_lm = J.T * J + u * np.eye(num_paras)
            delt = H_lm.I * g
            norm_2 = np.linalg.norm(delt)
            if norm_2 < eps:
                stop = True
            else:
                para_cur = para_past + delt.A   # 更新参数
                y_gj_cur = func(x, para_cur)
                J_cur = jacobi(x, para_cur)
                r_cur = np.matrix(y - y_gj_cur).T
                rou = ((np.linalg.norm(r_past) ** 2 - np.linalg.norm(r_cur) ** 2) / (delt.T.dot(u * delt + g))).A[0][0]
                if rou > 0:
                    para_past = para_cur
                    r_past = r_cur
                    J = jacobi(x, para_past)
                    g = J.T * r_past
                    stop  = (np.linalg.norm(g, ord=np.inf) <= eps) or (np.linalg.norm(r_past)**2 <= eps)
                    u = u * max(1 / 3, 1 - (2 * rou - 1) ** 3)
                    v = 2
                else:
                    u *= v
                    v *= 2
            if rou > 0 or stop:
                break
    return para_past
```

* 可视化

```
def show(x, y, para=None):
    figure = plt.figure()
    plt.scatter(x, y)

    if para is not None:
        x_ = np.arange(np.min(x), np.max(x), 0.05)
        y_ = para[0][0] * np.exp(para[1][0] / (x_ + para[2][0]))
        plt.plot(x_, y_, color='r')
    plt.show()
```

<img src="C:\Users\Voilencer\AppData\Roaming\Typora\typora-user-images\image-20210819203301261.png" alt="image-20210819203301261" style="zoom:67%;" />



# 参考文献

1. [LM](http://users.ics.forth.gr/~lourakis/levmar/levmar.pdf)

2. [LM(Levenberg–Marquardt)算法原理及其python自定义实现](https://www.pianshen.com/article/9422367969/)