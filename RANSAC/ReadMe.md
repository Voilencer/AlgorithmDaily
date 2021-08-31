[TOC] 

随机抽样一致算法(Random sample consensus, RANSAC)

***

# 一 理论介绍

## 简介	

通过反复选择数据中的一组随机子集，通过迭代方式来估计模型参数。

## 步骤

* step1：根据需求选择模型，随机选取N个点(>=模型最少点数)，对模型进行拟合；
* step2：设定容许误差sigma，在容差范围内的点为内点，在范围外的为外点，统计内点个数；
* step3：重新选取N个点，重复1~2步骤，直到达到设定的迭代次数；、
* step4：判断内点数最多的模型，即为最终的拟合结果；



## 特点

* 最小二乘法，对噪音大的数据集不适合，RANSAC可以选择部分子集来判断；

* 鲁棒的估计模型参数；
* RANSAC只有一定的概率得到可信的模型，概率与迭代次数成正比，如果设置迭代次数的上限，得到的结果可能不是最优的结果，甚至可能得到错误的结果；
* 要求设置跟问题相关的阀值；



# 二 直线拟合

1. 随机生成一组散点

```
X = range(0, 100, 10)
Y = [5 * _+  random.randint(0, 100) for _ in X]
```

2. 设置`RANSAC`参数

```
inner_thr = 10   # 阈值
iter_num = 100   # 迭代次数
model_num = 4   # 模型最小点数
```

3. `RANSAC`求解

```
def cal_model(X, Y):
    # y = k*x + b  最小二乘法
    Bval, Lval = [], []
    for _x, _y in zip(X, Y):
        Bval.append([_x, 1])
        Lval.append([_y])
    Bmat = np.matrix(Bval)
    Lmat = np.matrix(Lval)
    gRes = (Bmat.T * Bmat).I * (Bmat.T * Lmat)
    Vmat = Bmat * gRes - Lmat
    sigma = np.sqrt(Vmat.T * Vmat / (len(X) - 2))
    return gRes, sigma

def estimate_error(para, x, y):
    # kx+b-y
    y_gj = para[0][0] * x + para[1][0]
    return y_gj - y

def myRANSAC(X, Y, inner_thr, iter_num, model_num):
    npts = len(X)
    l_inner_num = []
    l_para, l_sigma = [], []
    for i in range(iter_num):
        inner_ind = []
        rand_ind = random.sample(range(npts), model_num)
        inner_ind.extend(rand_ind)
        not_rand_ind = [x for x in range(npts) if x not in rand_ind]
        ptsP_, ptsR_ = [], []
        for j in rand_ind:
            ptsP_.append(X[j])
            ptsR_.append(Y[j])
        para, sigma = cal_model(ptsP_, ptsR_)
        inner_num = 0
        for k in not_rand_ind:
            sigma = estimate_error(para, X[k], Y[k])
            if sigma[0][0] < inner_thr:
                inner_num += 1
                inner_ind.append(k)
        l_inner_num.append(inner_num)
        l_para.append(para)
        l_sigma.append(sigma)
    max_ind = np.argmax(l_inner_num)
    return l_para[max_ind].tolist(), l_sigma[max_ind]
```

4. 结果可视化

```
para, sigma =  myRANSAC(X, Y, inner_thr, iter_num, model_num)
print(para, sigma)

x1 = 0
y1 = para[0][0] * x1 + para[1][0]
x2 = 90
y2 = para[0][0] * x2 + para[1][0]

fig = plt.figure()
plt.scatter(X, Y)
plt.plot([x1, x2], [y1, y2], color='r')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title('RANSAC直线拟合')
plt.show()
```

<img src="C:\Users\Voilencer\AppData\Roaming\Typora\typora-user-images\image-20210808185957657.png" alt="image-20210808185957657" style="zoom:67%;" />

