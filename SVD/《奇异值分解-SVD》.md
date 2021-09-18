[TOC]

奇异值分解(Singular Value Decomposition，SVD)

***



# 特征值和特征向量

## 定义

$$
Ax = λx
$$

* 矩阵A是方阵， nxn
* 矩阵A的特征值λ， x是对应特征值λ的特征向量nx1

* 矩阵分解
  $$
  A = W\sum{}W^{-1}
  $$

  * W是由n个特征向量组成的nxn矩阵

  * $\sum{}$为n个特征值为主对角线的nxn维矩阵

  * 将W的n个特征向量标准化$||w_i||_2=1$，即$W^TW=I$，即$W^T = W^{-1}$
    $$
    A=W\sum{}W^T
    $$

## 求解

1. 转化为齐次线性方程$(A-λE)x=0$
2. $f(λ)=|A-λE|=0$方程的解空间为对应特征值λ的特征子空间
3. λ代入，求基础解系 为λ对应的特征向量                                                                                                                                                                                                                                                                                                                     



## python

* `np.linalg`
  * 计算特征值和特征向量
  * 特征值不会排序
  * V是归一化后的特征向量
  * W[i]和V[:,i]对应，  A * V[:,i] = W[i] * V[:,i]

```
assert A.shape[0] == A.shape[1]
det = np.linalg.det(A)
assert det != 0
W, V = np.linalg.eig(A)

# 验证Ax=lamdax
res1 = np.dot(A, Vt[:,0])
res2 = W[0] * Vt[:,0]

# 归一化的特征向量
res = np.dot(Vt[:,0], Vt[:,0].T)
print(res)
```



## SVD

* SVD可以对矩阵分解，不要求矩阵为方阵

$$
A = U\sum{}V^T
$$

* $A$是mxn矩阵
* $\sum$是mxn矩阵，除了对角线其余都是0，对角线的元素为奇异值
* $V$是nxn矩阵，单位正交矩阵$V^TV=I=U^TU$

<img src="C:\Users\Voilencer\AppData\Roaming\Typora\typora-user-images\image-20210918111324832.png" alt="image-20210918111324832" style="zoom:50%;" />

* 左奇异向量
  * 求解得到n个特征值和对应的n个特征向量，组成V

$$
(A^TA)v_i=λ_iv_i
$$

* 右奇异向量
  * 求解得到m个特征值和m个特征向量，组成U

$$
(AA^T)v_i=λ_iv_i
$$

* 奇异值

$$
A = U\sum{}V^T \\
AV = U\sum{} \\
Av_i = σ_iu_i \\
σ_i = Av_i / u_i
$$





## 性质及应用

## PCA特征降维

* 奇异值顺序排列，前10%奇异值之和占大多数比例，可以用最大的K个特征值和对应的左右奇异向量来近似描述矩阵

$$
A_{mxn} = U_{mxm}\sum{_{mxn}}V^T_{nxn} \\
≈U_{mxk}\sum{_{kxk}}V^T_{kxn}
$$



## 推荐系统

## 自然语言处理